"""

Access to the CalTech 101 Silhouettes (28*28) dataset.

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

class CalTech101Silhouettes(DataSet):
    def __init__(self, which_set='train', fname="caltech101_silhouettes_28_split1.mat", preproc=[]):
        super(CalTech101Silhouettes, self).__init__(preproc)

        _logger.info("Loading CalTech 101 Silhouettes data (28x28)")
        fname = datapath(fname)

        if which_set == 'train':
            self.X = scipy.io.loadmat(fname)['train_data'].astype(floatX)
        elif which_set == 'valid':
            self.X = scipy.io.loadmat(fname)['val_data'].astype(floatX)
        elif which_set == 'test':
            self.X = scipy.io.loadmat(fname)['test_data'].astype(floatX)
        else:
            raise ValueError("Unknown dataset %s" % which_set)


        self.n_datapoints = self.X.shape[0]

        self.Y = np.zeros((self.n_datapoints, 2), dtype=floatX)

