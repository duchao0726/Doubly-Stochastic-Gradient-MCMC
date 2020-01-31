#!/usr/bin/env python

from __future__ import division

import sys

import logging
from six import iteritems
from collections import OrderedDict
from time import time

import numpy as np

import theano
import theano.tensor as T


from training import TrainerBase

import pylab
from model import Model


class Reconstructor(TrainerBase):
    def __init__(self, **hyper_params):
        super(Reconstructor, self).__init__()

        self.register_hyper_param("figure_dir", default="", help="dir to save figure")
        self.register_hyper_param("iterations", default=100, help="number of iterations")
        self.register_hyper_param("invis_dims", default=np.arange(392,784), help="hollowed dimension")

        self.set_hyper_params(hyper_params)
        self.logger.debug("hyper_params %s" % self.get_hyper_params())

    def compile(self):
        """ Theano-compile neccessary functions """
        model = self.model

        assert isinstance(model, Model)

        model.setup()
        self.update_shvars()

        #---------------------------------------------------------------------
        self.logger.info("compiling ...")

        n_datapoints = self.dataset.n_datapoints

        X, _ = self.dataset.late_preproc(self.train_X, None)
        X_invis = X[:,self.invis_dims]

        # for simplicity only one sample is used
        X_new = model.reconstruct(X)

        updates = OrderedDict()
        updates[X] = T.set_subtensor(X_invis, X_new[:,self.invis_dims])

        self.reconstruct = theano.function(
                            inputs=[],
                            outputs=[],
                            updates=updates,
                            name="do_reconstruct")

    def perform_learning(self):
        self.update_shvars()

        model = self.model

        assert isinstance(model, Model)

        for i in range(self.iterations):
            self.reconstruct()
            self.logger.info("iteration %d .." % (i + 1))

        # prepare data for figure
        X_predict = self.train_X.get_value()
        X_origin = self.dataset.X
        X_hollow = np.copy(X_origin)
        X_hollow[:,self.invis_dims] = 0.7


        data_id = np.random.permutation(self.dataset.n_datapoints)
        # set figure info
        n_draw = 12
        transpose = False
        shape = (28,28)
        width = n_draw
        height = 3

        # draw
        pylab.figure(figsize=(16,4))
        for i in xrange(n_draw):
            pylab.subplot(height, width, i+1)
            if transpose:
                pylab.imshow( X_origin[data_id[i],:].reshape(shape).T, interpolation='nearest')
            else:
                pylab.imshow( X_origin[data_id[i],:].reshape(shape), interpolation='nearest')
            pylab.gray()
            pylab.axis('off')
            pylab.subplot(height, width, width+i+1)
            if transpose:
                pylab.imshow( X_hollow[data_id[i],:].reshape(shape).T, interpolation='nearest')
            else:
                pylab.imshow( X_hollow[data_id[i],:].reshape(shape), interpolation='nearest')
            pylab.gray()
            pylab.axis('off')
            pylab.subplot(height, width, 2*width+i+1)
            if transpose:
                pylab.imshow( X_predict[data_id[i],:].reshape(shape).T, interpolation='nearest')
            else:
                pylab.imshow( X_predict[data_id[i],:].reshape(shape), interpolation='nearest')
            pylab.gray()
            pylab.axis('off')


        pylab.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        pylab.subplots_adjust(wspace=0.05, hspace=0.05)

        figname = self.figure_dir + 'miss.pdf'
        pylab.savefig(figname)

