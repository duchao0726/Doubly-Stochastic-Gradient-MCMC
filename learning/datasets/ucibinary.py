"""

Access to the UCI binary dataset.

"""

from __future__ import division

import os
import logging
import cPickle as pickle
import gzip
import h5py

import numpy as np

import theano
import theano.tensor as T

from learning.datasets import DataSet, datapath

from learning.setting import *

_logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------

class UCIBinary(DataSet):
    def __init__(self, data_name, which_set='train', preproc=[]):
        super(UCIBinary, self).__init__(preproc)

        UCIdatasets = {
                'adult': 'adult.h5',
                'connect4': 'connect4.h5',
                'dna': 'dna.h5',
                'mushrooms': 'mushrooms.h5',
                'nips': 'nips.h5',
                'ocrletters': 'ocr_letters.h5',
                'rcv1': 'rcv1.h5',
                'web': 'web.h5'
            }
        assert data_name in UCIdatasets.keys()

        _logger.info("Loading %s data" % data_name)
        fname = datapath(UCIdatasets[data_name])

        try:
            with h5py.File(fname, "r") as h5:

                if which_set == 'train':
                    train_x = h5['train']
                    self.X = np.array(train_x).astype(floatX)
                elif which_set == 'valid':
                    valid_x = h5['valid']
                    self.X = np.array(valid_x).astype(floatX)
                elif which_set == 'test':
                    test_x = h5['test']
                    self.X = np.array(test_x).astype(floatX)
                else:
                    raise ValueError("Unknown dataset %s" % which_set)

        except KeyError, e:
            logger.info("Failed to read data from %s: %s" % (fname, e))
            exit(1)

        except IOError, e:
            logger.info("Failed to open %s: %s" % (fname, e))
            exit(1)

        self.n_datapoints = self.X.shape[0]

        self.Y = np.zeros((self.n_datapoints, 2), dtype=floatX)

