#!/usr/bin/env python

from __future__ import division

import logging

import numpy as np

import theano
import theano.tensor as T
from theano.printing import Print

from learning.models.module import TopModule, Module

import nnet

from learning.setting import *

_logger = logging.getLogger(__name__)

log2pi = T.constant(np.log(2*np.pi).astype(theano.config.floatX))

#----------------------------------------------------------------------------

class UnitGaussianSampler:
    def __init__(self):
        self.params = []

    def samplesIshape(self, shape):
        return theano_rng.normal(shape)

    def log_likelihood_samples(self, samples):
        '''Given samples as rows of a matrix, returns their log-likelihood under the zero mean unit covariance Gaussian as a vector'''
        return -log2pi*T.cast(samples.shape[1], floatX)/2 - T.sum(T.sqr(samples), axis=1) / 2


class GaussianSampler:
    def __init__(self, h_network, mean_network, sigma_network):
        self.h_network = h_network
        self.mean_network = mean_network
        self.sigma_network = sigma_network

        self.params = self.h_network.params + self.mean_network.params + self.sigma_network.params

    def mean_sigmaIx(self, x):
        '''Returns the mean and the square root of the covariance of the Gaussian'''
        h = self.h_network.yIx(x)
        mean = self.mean_network.yIx(h)
        sigma = self.sigma_network.yIx(h)

        return mean, sigma

    def meanIx(self, x):
        h = self.h_network.yIx(x)
        mean = self.mean_network.yIx(h)
        return mean

    def samplesImean_sigma(self, mean, sigma):
        unit_gaussian_samples = theano_rng.normal(mean.shape)
        return sigma * unit_gaussian_samples + mean

    def samplesIx(self, x):
        mean, sigma = self.mean_sigmaIx(x)
        return self.samplesImean_sigma(mean, sigma)

    def log_likelihood_samplesImean_sigma(self, samples, mean, sigma):
        return -log2pi*T.cast(samples.shape[1], floatX) / 2 -                \
               T.sum(T.sqr((samples-mean)/sigma) + 2*T.log(sigma), axis=1) / 2

    def log_likelihood_samplesIx(self, samples, x):
        mean, sigma = self.mean_sigmaIx(x)
        return self.log_likelihood_samplesImean_sigma(samples, mean, sigma)

    def first_linear_layer_weights_np(self):
        return self.h_network.first_linear_layer_weights_np()

    @staticmethod
    def random(n_units, mean=None):
        h_network = nnet.random_linear_then_tanh_chain(n_units[:-1])
        mean_network = nnet.Linear.random(n_units[-2], n_units[-1])
        if mean is not None:
            mean_network.b.set_value(mean.astype(floatX))
        sigma_network = nnet.NNet().add_layer(nnet.Linear.random(n_units[-2], n_units[-1])).add_layer(nnet.Exponential())

        return GaussianSampler(h_network, mean_network, sigma_network)


class BernoulliSampler:
    def __init__(self, mean_network):
        self.mean_network = mean_network

        self.params = self.mean_network.params

    def meanIx(self, x):
        return self.mean_network.yIx(x)

    def samplesImean(self, mean):
        return T.cast(T.le(theano_rng.uniform(mean.shape), mean), mean.dtype)

    def samplesIx(self, x):
        return self.samplesImean(self.meanIx(x))

    def log_likelihood_samplesImean(self, samples, mean):
        return T.sum(samples * T.log(mean) + (1 - samples) * T.log(1 - mean), axis=1)

    def log_likelihood_samplesIx(self, samples, x):
        mean = self.meanIx(x)
        return self.log_likelihood_samplesImean(samples, mean)

    def last_linear_layer_weights_np(self):
        return self.mean_network.last_linear_layer_weights_np()

    def first_linear_layer_weights_np(self):
        return self.mean_network.first_linear_layer_weights_np()

    @staticmethod
    def random(n_units, bias=None):
        mean_network = nnet.random_linear_then_tanh_chain(n_units[:-1])

        mean_network.add_layer(nnet.Linear.random(n_units[-2], n_units[-1]))

        if bias is not None:
            mean_network.layers[-1].b.set_value(bias.astype(theano.config.floatX))

        mean_network.add_layer(nnet.Sigmoid())

        return BernoulliSampler(mean_network)



#=============================================================================


class StandardVAETop(TopModule):
    """ Factoized standard normal top layer """
    def __init__(self, **hyper_params):
        super(StandardVAETop, self).__init__()

        # Hyper parameters
        self.register_hyper_param('n_X', help='no. normal variables')

        # No Model parameters

        self.set_hyper_params(hyper_params)

        # wrap Burda's prior sampler
        self.sampler = UnitGaussianSampler()

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

        # Calculate log-normal
        log_prob = self.sampler.log_likelihood_samples(X)

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

        X = self.sampler.samplesIshape((n_samples, n_X))

        return X, self.log_prob(X)

#----------------------------------------------------------------------------

class VAE(Module):
    """ Variational Auto-Encoder Layer """
    def __init__(self, **hyper_params):
        super(VAE, self).__init__()

        self.vaelogger = logging.getLogger('VAE')

        self.register_hyper_param('n_X', help='no. bottom-layer variables')
        self.register_hyper_param('n_Y', help='no. top-layer variables')
        self.register_hyper_param('det_units', help='# units of deterministic layers', default=[])
        self.register_hyper_param('data_type', help='distribution of X', default="binary")
        self.register_hyper_param('bias', help='bias of data X', default=None)
        self.register_hyper_param('mean', help='mean of data X', default=None)

        self.set_hyper_params(hyper_params)

        n_X, n_Y, det_units, data_type = self.get_hyper_params(['n_X', 'n_Y', 'det_units', 'data_type'])
        # print n_X, n_Y, det_units, data_type

        self.sampler = None

        if data_type == 'binary':
            self.sampler = BernoulliSampler.random([n_Y]+det_units+[n_X], bias=self.bias)

            self.register_model_param('mean_b', help='mean bias of bottom-layer',
                    default=self.sampler.mean_network.params[-1])
            self.register_model_param('mean_w', help='mean weights of bottom-layer',
                    default=self.sampler.mean_network.params[-2])
            for i in range(-3, -len(self.sampler.mean_network.params)-1, -1):
                self.register_model_param('det_param_'+str(-i-3),
                        help='weights and bias of deterministic layers',
                        default=self.sampler.mean_network.params[i])

        elif data_type == 'continuous':
            self.sampler = GaussianSampler.random([n_Y]+det_units+[n_X], mean=self.mean)

            self.register_model_param('mean_b', help='mean bias of bottom-layer',
                    default=self.sampler.mean_network.params[-1])
            self.register_model_param('mean_w', help='mean weights of bottom-layer',
                    default=self.sampler.mean_network.params[-2])
            self.register_model_param('sigma_b', help='sigma bias of bottom-layer',
                    default=self.sampler.sigma_network.params[-1])
            self.register_model_param('sigma_w', help='sigma weights of bottom-layer',
                    default=self.sampler.sigma_network.params[-2])
            for i in range(len(self.sampler.h_network.params)):
                self.register_model_param('det_param_'+str(len(self.sampler.h_network.params)-i-1),
                        help='weights and bias of deterministic layers',
                        default=self.sampler.h_network.params[i])

        assert self.sampler is not None

        # Variational Auto-Encoder Layer

        self.vaelogger.info(str([n_Y]+det_units+[n_X]))


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

        log_prob = self.sampler.log_likelihood_samplesIx(X, Y)

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

        # sample X given Y
        X = self.sampler.samplesIx(Y)
        log_prob = self.sampler.log_likelihood_samplesIx(X, Y)

        return X, log_prob

    def sample_expected(self, Y):
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
        prob_X = self.sampler.meanIx(Y)

        return prob_X


