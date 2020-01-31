#!/usr/bin/env python

from __future__ import division

import logging

import numpy as np

import theano
import theano.tensor as T
from theano.printing import Print

from learning.model import default_weights
from learning.models.module import TopModule, Module, GibbsModule

from learning.utils.unrolled_scan import unrolled_scan

from learning.setting import *

_logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------

class GibbsSBNTop(TopModule):
    """ FactoizedBernoulliTop top layer """
    def __init__(self, **hyper_params):
        super(GibbsSBNTop, self).__init__()

        # Hyper parameters
        self.register_hyper_param('n_X', help='no. binary variables')

        # Model parameters
        self.register_model_param('a', help='sigmoid(a) prior',
            default=lambda: -np.ones(self.n_X))

        self.set_hyper_params(hyper_params)

    def log_prob(self, X):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        X:      T.tensor
            samples from X

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        a, = self.get_model_params(['a'])

        # Calculate log-bernoulli
        prob_X = self.sigmoid(a)
        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = log_prob.sum(axis=1)

        return log_prob

    def sample(self, n_samples):
        """ Sample from this toplevel module and return X ~ P(X), log(P(X))

        Parameters
        ----------
        n_samples:
            number of samples to drawn

        Returns
        -------
        X:      T.tensor
            samples from this module
        log_p:  T.tensor
            log-probabilities for the samples returned in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        a, = self.get_model_params(['a'])

        # sample hiddens
        prob_X = self.sigmoid(a)
        U = theano_rng.uniform((n_samples, n_X), nstreams=512)
        X = T.cast(U <= prob_X, dtype=floatX)

        return X, self.log_prob(X)

    def sample_expected(self):
        n_X, = self.get_hyper_params(['n_X'])
        a, = self.get_model_params(['a'])

        # Calculate log-bernoulli
        prob_X = self.sigmoid(a)

        return prob_X

    def calNegEnergy(self, X, rate=1.0):
        a, = self.get_model_params(['a'])
        a = a * rate

        nE = T.sum(X*a - T.log(1+T.exp(a)), axis=1)
        return nE

#----------------------------------------------------------------------------

class GibbsSBN(GibbsModule):
    """ SigmoidBeliefLayer """
    def __init__(self, **hyper_params):
        super(GibbsSBN, self).__init__()

        # Hyper parameters
        self.register_hyper_param('n_X', help='no. lower-layer binary variables')
        self.register_hyper_param('n_Y', help='no. upper-layer binary variables')
        self.register_hyper_param('unroll_scan', default=1)

        # Model parameters
        self.register_model_param('b', help='P lower-layer bias', default=lambda: -np.ones(self.n_X))
        self.register_model_param('W', help='P weights', default=lambda: default_weights(self.n_Y, self.n_X) )

        self.set_hyper_params(hyper_params)

    def log_prob(self, X, Y):
        """ Evaluate the log-probability for the given samples.

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer
        X:      T.tensor
            samples from the lower layer

        Returns
        -------
        log_p:  T.tensor
            log-probabilities for the samples in X and Y
        """
        W, b = self.get_model_params(['W', 'b'])

        # posterior P(X|Y)
        prob_X = self.sigmoid(T.dot(Y, W) + b)
        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = T.sum(log_prob, axis=1)

        return log_prob

    def sample(self, Y):
        """ Given samples from the upper layer Y, sample values from X
            and return then together with their log probability.

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer

        Returns
        -------
        X:      T.tensor
            samples from the lower layer
        log_p:  T.tensor
            log-posterior for the samples returned in X
        """
        n_X, = self.get_hyper_params(['n_X'])
        W, b = self.get_model_params(['W', 'b'])

        n_samples = Y.shape[0]

        # sample X given Y
        prob_X = self.sigmoid(T.dot(Y, W) + b)
        U = theano_rng.uniform((n_samples, n_X), nstreams=512)
        X = T.cast(U <= prob_X, dtype=floatX)

        log_prob = X*T.log(prob_X) + (1-X)*T.log(1-prob_X)
        log_prob = log_prob.sum(axis=1)

        return X, log_prob

    def sample_expected(self, Y, rate=1.0):
        """ Given samples from the upper layer Y, return
            the probability for the individual X elements

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer

        Returns
        -------
        X:      T.tensor
        """
        W, b = self.get_model_params(['W', 'b'])
        W = W * rate
        b = b * rate

        prob_X = self.sigmoid(T.dot(Y, W) + b)

        return prob_X

    def post_sample(self, X, Y_init=None, prior=None, rate=1.0):
        """ posterior sampling """
        n_X, n_Y = self.get_hyper_params(['n_X', 'n_Y'])
        W, b = self.get_model_params(['W', 'b'])
        W = W * rate
        b = b * rate

        batch_size = X.shape[0]

        if Y_init is None:
            Y_init = T.cast(theano_rng.binomial((batch_size, W.shape[0]), nstreams=512), dtype=floatX)
        if prior is None:
            prior = T.zeros(Y_init.shape, dtype=floatX)

        WYb_init = T.dot(Y_init, W) + b
        rand = theano_rng.uniform((n_Y, batch_size), nstreams=512)

        def one_iter(k, Wk, prior_k, rand_k, WYb, Y, X):
            WXb = T.dot(X,Wk)+prior_k
            WYb = WYb - T.outer(Y[:,k],Wk)
            reg_0 = - T.sum(T.nnet.softplus(WYb), axis=1)
            reg_1 = - T.sum(T.nnet.softplus(WYb+Wk), axis=1) + WXb
            prob = 1. / (1 + T.exp(reg_0 - reg_1))
            Y = T.set_subtensor(Y[:,k], T.cast(rand_k <= prob, dtype=floatX))
            WYb = WYb + T.outer(Y[:,k],Wk)
            return WYb, Y

        [WYb_list, Y_list], updates = unrolled_scan(
                    fn = one_iter,
                    sequences=[T.arange(n_Y), W, prior.T, rand],
                    outputs_info=[WYb_init, Y_init],
                    non_sequences=[X],
                    unroll=self.unroll_scan
                )

        return Y_list[-1]

    def calNegEnergy(self, X, Y, rate=1.0):
        W, b = self.get_model_params(['W', 'b'])
        W = W * rate
        b = b * rate

        WYb = T.dot(Y, W) + b
        nE = T.sum(X*WYb - T.log(1+T.exp(WYb)), axis=1)
        return nE

