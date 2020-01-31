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
from learning.utils.datalog import dlog
from learning.utils.misc import *

from learning.models.module import TopModule, Module

_logger = logging.getLogger(__name__)


#=============================================================================

class VAEDGMLayerStack(Model):
    def __init__(self, **hyper_params):
        super(VAEDGMLayerStack, self).__init__()

        # Hyper parameters
        self.register_hyper_param('p_layers', help='STBP P layers', default=[])
        self.register_hyper_param('q_layers', help='STBP Q layers', default=[])
        self.register_hyper_param('n_samples', help='no. of samples to use', default=10)
        self.register_hyper_param('nu', help='mu of student-t', default=2.2)
        self.register_hyper_param('sigma', help='sigma of student-t', default=0.0953471)

        self.set_hyper_params(hyper_params)

    def setup(self):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        assert len(p_layers) == len(q_layers)+1
        assert isinstance(p_layers[-1], TopModule)


        self.n_X = p_layers[0].n_X

        for l in xrange(0, n_layers-1):
            assert isinstance(p_layers[l], Module)
            assert isinstance(q_layers[l], Module)
            assert p_layers[l].n_Y == p_layers[l+1].n_X
            assert p_layers[l].n_Y == q_layers[l].n_X

            p_layers[l].setup()
            q_layers[l].setup()


        p_layers[-1].setup()

    def sample_p(self, n_samples):
        """ Draw *n_samples* drawn from the P-model.

        This method returns a list with the samples values on all layers
        and the correesponding log_p.
        """
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        # Generate samples from the generative model
        samples = [None]*n_layers

        samples[-1], log_prob = p_layers[-1].sample(n_samples)
        for l in xrange(n_layers-1, 0, -1):
            samples[l-1], log_p_l = p_layers[l-1].sample(samples[l])
            log_prob += log_p_l

        return samples, log_prob

    def sample_q(self, X, Y=None):
        """ Given a set of observed X, samples from q(H | X) and calculate
            both P(X, H) and Q(H | X)
        """
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        size = X.shape[0]

        # Prepare input for layers
        samples = [None]*n_layers
        log_q   = [None]*n_layers
        log_p   = [None]*n_layers

        samples[0] = X
        log_q[0]   = T.zeros([size])

        # Generate samples (feed-forward)
        for l in xrange(n_layers-1):
            samples[l+1], log_q[l+1] = q_layers[l].sample(samples[l])

        # Get log_probs from generative model
        log_p[n_layers-1] = p_layers[n_layers-1].log_prob(samples[n_layers-1])
        for l in xrange(n_layers-1, 0, -1):
            log_p[l-1] = p_layers[l-1].log_prob(samples[l-1], samples[l])

        return samples, log_p, log_q

    def log_likelihood(self, X, Y=None, n_samples=None):
        p_layers = self.p_layers
        q_layers = self.q_layers
        n_layers = len(p_layers)

        if n_samples == None:
            n_samples = self.n_samples

        batch_size = X.shape[0]

        # Get samples
        X = f_replicate_batch(X, n_samples)
        samples, log_p, log_q = self.sample_q(X, None)

        # Reshape and sum
        log_p_all = T.zeros((batch_size, n_samples))
        log_q_all = T.zeros((batch_size, n_samples))
        for l in xrange(n_layers):
            samples[l] = samples[l].reshape((batch_size, n_samples, p_layers[l].n_X))
            log_q[l] = log_q[l].reshape((batch_size, n_samples))
            log_p[l] = log_p[l].reshape((batch_size, n_samples))
            log_p_all += log_p[l]   # agregate all layers
            log_q_all += log_q[l]   # agregate all layers

        # Approximate log P(X)
        log_px = f_logsumexp(log_p_all-log_q_all, axis=1) - T.log(n_samples)

        # Calculate samplig weights
        log_ws_matrix = log_p_all - log_q_all
        log_ws_minus_max = log_ws_matrix - T.max(log_ws_matrix, axis=1, keepdims=True)
        ws = T.exp(log_ws_minus_max)
        w = ws / T.sum(ws, axis=1, keepdims=True) # ws_normalized
        log_w = T.log(w)

        # Calculate KL(P|Q), Hp, Hq
        KL = [None]*n_layers
        Hp = [None]*n_layers
        Hq = [None]*n_layers
        for l in xrange(n_layers):
            KL[l] = T.sum(w*(log_p[l]-log_q[l]), axis=1)
            Hp[l] = f_logsumexp(log_w+log_p[l], axis=1)
            Hq[l] = T.sum(w*log_q[l], axis=1)

        return log_px, w, log_p_all, log_q_all, KL, Hp, Hq

    def get_gradients(self, X, Y, n_samples, layer_discount=1.0):
        """ return log_PX and two OrderedDict with parameter gradients,
            gradients of p model : estimation of log p(X_batch),
            gradients of q model : L_s estimation of log p(X_batch)
        """
        nu = self.nu
        sigma = self.sigma

        log_PX, w, log_p, log_q, KL, Hp, Hq = self.log_likelihood(X, Y, n_samples=n_samples)

        log_ws_matrix = log_p - log_q

        batch_log_PX = T.sum(log_PX)

        log_like = T.sum(T.sum(log_p*w, axis=1))
        neglog_like = - log_like
        log_px_s = T.sum(T.sum(log_ws_matrix*w, axis=1))

        grads_p = OrderedDict()
        for nl, layer in enumerate(self.p_layers):
            for name, shvar in iteritems(layer.get_model_params()):
                grad_p = (layer_discount ** nl) * T.grad(neglog_like, shvar, consider_constant=[w])

                shvar_logprior = - 0.5 * (1 + nu) * T.sum(T.log(1 + shvar**2 / (nu * sigma * sigma)))
                shvar_neglogprior = - shvar_logprior
                grad_p_prior = (layer_discount ** nl) * T.grad(shvar_neglogprior, shvar)
                grads_p[shvar] = (grad_p, grad_p_prior)

        grads_q = OrderedDict()
        for nl, layer in enumerate(self.q_layers):
            for name, shvar in iteritems(layer.get_model_params()):
                grads_q[shvar] = (layer_discount ** nl) * T.grad(log_px_s, shvar, consider_constant=[w])

        return batch_log_PX, grads_p, grads_q

    def reconstruct(self, X):
        """ Given a set of observed X, and invisible dimentions, predicte ivisible dimentions of X
        """
        p_layers = self.p_layers
        q_layers = self.q_layers

        size = X.shape[0]

        # Prepare input for layers
        samples = [None]*2

        # for simplicity only one sample is used

        # Generate samples (feed-forward)
        samples[1], _ = q_layers[0].sample(X)
        # Get samples from generative model
        samples[0], _ = p_layers[0].sample(samples[1])

        return samples[0]

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
