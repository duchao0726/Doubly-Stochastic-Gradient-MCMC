"""

Access to the OMNIGLOT (28*28) dataset.

"""

from __future__ import division

import os
import logging
import cPickle as pickle
import gzip

import numpy as np
import scipy.io

import theano
import theano.tensor as T

from learning.datasets import DataSet, datapath

from learning.setting import *

_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------

class Omniglot(DataSet):
    def __init__(self, which_set='train', fname="chardata.mat", shuffle_seed=123, n_used_for_validation=1345, preproc=[]):
        super(Omniglot, self).__init__(preproc)

        def reshape_data(data):
            return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')

        _logger.info("Loading Omniglot data (28x28)")
        fname = datapath(fname)

        omni_raw = scipy.io.loadmat(fname)

        train_data = reshape_data(omni_raw['data'].T.astype('float32'))
        test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))

        permutation = np.random.RandomState(seed=shuffle_seed).permutation(train_data.shape[0])
        train_data = train_data[permutation]


        if which_set == 'train':
            self.X = train_data[:-n_used_for_validation]
        elif which_set == 'valid':
            self.X = train_data[-n_used_for_validation:]
        elif which_set == 'test':
            self.X = test_data
        else:
            raise ValueError("Unknown dataset %s" % which_set)


        self.n_datapoints = self.X.shape[0]

        self.Y = np.zeros((self.n_datapoints, 2), dtype=floatX)

