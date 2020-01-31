#!/usr/bin/env python

from __future__ import division

import logging
from six import iteritems
from abc import ABCMeta, abstractmethod

import numpy as np

import theano
import theano.tensor as T

from learning.model import Model

from learning.setting import *

#=============================================================================

class TopModule(Model):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(TopModule, self).__init__()
        self.register_hyper_param('clamp_sigmoid', default=False)

    def setup(self):
        pass

    def sigmoid(self, x):
        """ Compute the element wise sigmoid function of x

        Depending on the *clamp_sigmoid* hyperparameter, this might
        return a saturated sigmoid T.nnet.sigmoid(x)*0.9999 + 0.000005
        """
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    @abstractmethod
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
        return X, log_p

    @abstractmethod
    def log_prob(self, X):
        """ Calculate the log-probabilities for the samples in X

        Parameters
        ----------
        X:      T.tensor
            samples to evaluate

        Returns
        -------
        log_p:  T.tensor
        """
        return log_p

class Module(Model):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Module, self).__init__()

        self.register_hyper_param('clamp_sigmoid', default=False)

    def setup(self):
        pass

    def sigmoid(self, x):
        """ Compute the element wise sigmoid function of x

        Depending on the *clamp_sigmoid* hyperparameter, this might
        return a saturated sigmoid T.nnet.sigmoid(x)*0.9999 + 0.000005
        """
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    @abstractmethod
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
        X, log_p = None, None
        return X, log_p

    @abstractmethod
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
        return log_p

#=============================================================================

class GibbsModule(Model):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(GibbsModule, self).__init__()

        self.register_hyper_param('clamp_sigmoid', default=False)

    def setup(self):
        pass

    def sigmoid(self, x):
        """ Compute the element wise sigmoid function of x

        Depending on the *clamp_sigmoid* hyperparameter, this might
        return a saturated sigmoid T.nnet.sigmoid(x)*0.9999 + 0.000005
        """
        if self.clamp_sigmoid:
            return T.nnet.sigmoid(x)*0.9999 + 0.000005
        else:
            return T.nnet.sigmoid(x)

    @abstractmethod
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
        X, log_p = None, None
        return X, log_p

    @abstractmethod
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
        log_p = None
        return log_p

    @abstractmethod
    def sample_expected(self, Y):
        """ Given samples from the upper layer Y, return
            the probability for the individual X elements

        Parameters
        ----------
        Y:      T.tensor
            samples from the upper layer

        Returns
        -------
        exp:    T.tensor
        """
        exp = None
        return exp

    @abstractmethod
    def post_sample(self, X, Y_init=None, prior=None):
        """ Posterior sample for the hidden hidden layer

        Parameters
        ----------
        X:      T.tensor
            samples of this layer

        prior:  T.tensor
            prior from upper layer

        Returns
        -------
        Y:      T.tensor
        """
        Y_post = None
        return Y_post

