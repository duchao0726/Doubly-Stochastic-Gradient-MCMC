#!/usr/bin/env python

from __future__ import division

import logging
from six import iteritems
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import numpy as np

import theano
import theano.tensor as T
from theano.printing import Print

from learning.model import Model
from learning.utils.datalog  import dlog
from learning.utils.misc import *
from learning.utils.unrolled_scan import unrolled_scan

from learning.models.module import TopModule, Module, GibbsModule

from learning.setting import *

_logger = logging.getLogger(__name__)

#=============================================================================

class GibbsDGMLayerStack(Model):
    def __init__(self, **hyper_params):
        super(GibbsDGMLayerStack, self).__init__()

        # Hyper parameters
        self.register_hyper_param('p_layers', help='model layers', default=[])
        self.register_hyper_param('q_layers', help='should be empty', default=[])
        self.register_hyper_param('n_samples', help='no. of samples to use', default=10)
        self.register_hyper_param('nu', help='mu of student-t', default=2.2)
        self.register_hyper_param('sigma', help='sigma of student-t', default=0.0953471)

        self.set_hyper_params(hyper_params)

    def setup(self):
        p_layers = self.p_layers
        n_layers = len(p_layers)

        assert len(self.q_layers) == 0
        assert isinstance(p_layers[-1], TopModule)


        self.n_X = p_layers[0].n_X

        for l in xrange(0, n_layers-1):
            assert isinstance(p_layers[l], GibbsModule)
            assert p_layers[l].n_Y == p_layers[l+1].n_X

            p_layers[l].setup()

        p_layers[-1].setup()

    def get_hidden_dim(self):
        p_layers = self.p_layers
        n_layers = len(p_layers)

        num_total_hidden = 0
        dim = [None] * n_layers
        dim[0] = 0

        for l in range(n_layers - 1):
            dim[l+1] = dim[l] + p_layers[l].n_Y

        sum_hidden_dim = dim[-1]

        return sum_hidden_dim, dim

    def sample_p(self, n_samples):
        """ Draw *n_samples* drawn from the P-model.

        This method returns a list with the samples values on all layers
        and the correesponding log_p.
        """
        p_layers = self.p_layers
        n_layers = len(p_layers)

        # Generate samples from the generative model
        samples = [None]*n_layers

        samples[-1], log_prob = p_layers[-1].sample(n_samples)
        for l in xrange(n_layers-1, 0, -1):
            samples[l-1], log_p_l = p_layers[l-1].sample(samples[l])
            log_prob += log_p_l

        return samples, log_prob

    def init_hidden(self, X, n_samples):
        """ Given a set of observed X, initialize the corresponding hidden variables
        """
        p_layers = self.p_layers
        n_layers = len(p_layers)

        X = f_replicate_batch(X, n_samples)

        size = X.shape[0]

        # initialize samples from marginal posterior distribution
        samples = [None]*n_layers
        samples[0] = X
        for l in xrange(n_layers-1):
            samples[l+1] = p_layers[l].post_sample(samples[l])

        return T.concatenate(samples[1::], axis=1)


    def sample_hidden(self, X, H, burnin=1):
        """ Given a set of observed X, samples from the posterior p(H | X) and calculate P(X, H)
        """
        p_layers = self.p_layers
        n_layers = len(p_layers)

        size = X.shape[0]

        samples = [X]

        _, dim = self.get_hidden_dim()
        for l in range(len(dim)-1):
            samples = samples + [H[:,dim[l]:dim[l+1]]]

        assert n_layers == len(samples)

        for it in range(burnin):
            for l in xrange(n_layers-2):
                samples[l+1] = p_layers[l].post_sample(samples[l], Y_init=samples[l+1], prior=p_layers[l+1].sample_expected(samples[l+2]))
            samples[n_layers-1] = p_layers[n_layers-2].post_sample(samples[n_layers-2], Y_init=samples[n_layers-1], prior=p_layers[n_layers-1].sample_expected())

        # Get log_probs from generative model
        log_p = [None]*n_layers

        log_p[n_layers-1] = p_layers[n_layers-1].log_prob(samples[n_layers-1])
        for l in xrange(n_layers-1, 0, -1):
            log_p[l-1] = p_layers[l-1].log_prob(samples[l-1], samples[l])

        return samples, log_p

    def log_likelihood(self, X, H, n_samples):
        p_layers = self.p_layers
        n_layers = len(p_layers)

        batch_size = X.shape[0]

        # Get samples
        X = f_replicate_batch(X, n_samples)

        samples, log_p = self.sample_hidden(X, H)

        # Reshape and sum
        log_p_all = T.zeros((batch_size, n_samples))
        for l in xrange(n_layers):
            log_p[l] = log_p[l].reshape((batch_size, n_samples))
            log_p_all += log_p[l]   # agregate all layers

        log_xlike = T.mean(log_p[0], axis=1)

        return log_xlike, log_p_all, samples

    def get_gradients(self, X, H, n_samples, layer_discount=1.0):
        """ return log_p(x|z) and an OrderedDict with parameter gradients
        """
        nu = self.nu
        sigma = self.sigma

        log_xlike, log_p_all, samples = self.log_likelihood(X, H, n_samples)

        batch_log_Pxz = T.sum(log_xlike)

        log_like = T.sum(T.mean(log_p_all, axis=1))
        neglog_like = - log_like

        grads_p = OrderedDict()
        for nl, layer in enumerate(self.p_layers):
            for name, shvar in iteritems(layer.get_model_params()):
                grad_p = (layer_discount ** nl) * T.grad(neglog_like, shvar, consider_constant=samples)

                shvar_logprior = - 0.5 * (1 + nu) * T.sum(T.log(1 + shvar**2 / (nu * sigma * sigma)))
                shvar_neglogprior = - shvar_logprior
                grad_p_prior = (layer_discount ** nl) * T.grad(shvar_neglogprior, shvar)

                grads_p[shvar] = (grad_p, grad_p_prior)

        return batch_log_Pxz, grads_p, T.concatenate(samples[1::], axis=1)

    #------------------------------------------------------------------------
    def get_p_params(self):
        params = OrderedDict()
        for l in self.p_layers:
            params.update( l.get_model_params() )
        return params

    def get_q_params(self):
        params = OrderedDict()
        for l in self.q_layers:
            params.update( l.get_model_params() )
        return params

    def model_params_to_dict(self):
        vals = {}
        for n,l in enumerate(self.p_layers):
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.P.%s" % (n, pname)
                vals[key] = shvar.get_value()
        for n,l in enumerate(self.q_layers):
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.Q.%s" % (n, pname)
                vals[key] = shvar.get_value()
        return vals

    def model_params_from_dict(self, vals):
        for n,l in enumerate(self.p_layers):
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.P.%s" % (n, pname)
                value = vals[key]
                shvar.set_value(value)
        for n,l in enumerate(self.q_layers):
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.Q.%s" % (n, pname)
                value = vals[key]
                shvar.set_value(value)

    def model_params_to_dlog(self, dlog):
        vals = self.model_params_to_dict()
        dlog.append_all(vals)

    def model_params_from_dlog(self, dlog, row=-1):
        for n,l in enumerate(self.p_layers):
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.P.%s" % (n, pname)
                value = dlog.load(key)
                shvar.set_value(value)
        for n,l in enumerate(self.q_layers):
            for pname, shvar in iteritems(l.get_model_params()):
                key = "L%d.Q.%s" % (n, pname)
                value = dlog.load(key)
                shvar.set_value(value)

    def model_params_from_h5(self, h5, row=-1, basekey="model."):
        for n,l in enumerate(self.p_layers):
            try:
                for pname, shvar in iteritems(l.get_model_params()):
                    key = "%sL%d.P.%s" % (basekey, n, pname)
                    value = h5[key][row]
                    shvar.set_value(value)
            except KeyError:
                if n >= len(self.p_layers)-2:
                    _logger.warning("Unable to load top P-layer params %s[%d]... continuing" % (key, row))
                    continue
                else:
                    _logger.error("Unable to load %s[%d] from %s" % (key, row, h5.filename))
                    raise

        for n,l in enumerate(self.q_layers):
            try:
                for pname, shvar in iteritems(l.get_model_params()):
                    key = "%sL%d.Q.%s" % (basekey, n, pname)
                    value = h5[key][row]
                    shvar.set_value(value)
            except KeyError:
                if n == len(self.q_layers)-1:
                    _logger.warning("Unable to load top Q-layer params %s[%d]... continuing" % (key, row))
                    continue
                _logger.error("Unable to load %s[%d] from %s" % (key, row, h5.filename))
                raise
