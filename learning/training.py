#!/usr/bin/env python

from __future__ import division

import sys

import abc
import logging
from six import iteritems
from collections import OrderedDict
from time import time

import math
import numpy as np
import progressbar as pbar

import theano
import theano.tensor as T

import utils.datalog as dlog

from hyperbase import HyperBase
from termination import Termination
from dataset import DataSet
from model import Model

from learning.setting import *

#=============================================================================
def get_adam_updates(grads, learning_rate=3e-4, decay1=0.1, decay2=0.001, weight_decay=0.0):

    updates = OrderedDict()

    it = theano.shared(np.asarray(0., dtype=floatX), name='it', borrow=False)
    updates[it] = it + 1.

    fix1 = 1.-(1.-decay1)**(it+1.) # To make estimates unbiased
    fix2 = 1.-(1.-decay2)**(it+1.) # To make estimates unbiased
    lr_t = learning_rate * T.sqrt(fix2) / fix1

    for shvar, grad in iteritems(grads):

        if weight_decay > 0:
            grad -= weight_decay * shvar #T.tanh(w[i])

        # mean_squared_grad := E[g^2]_{t-1}
        mom1 = theano.shared(shvar.get_value() * 0., name=("%s_mom1"%shvar.name), borrow=False)
        mom2 = theano.shared(shvar.get_value() * 0., name=("%s_mom2"%shvar.name), borrow=False)

        # Update moments
        mom1_new_temp = mom1 + decay1 * (grad - mom1)
        mom2_new_temp = mom2 + decay2 * (T.sqr(grad) - mom2)
        mom1_new = T.switch(T.isnan(mom1_new_temp), mom1, mom1_new_temp)
        mom2_new = T.switch(T.isnan(mom2_new_temp), mom2, mom2_new_temp)

        # Compute the effective gradient and effective learning rate
        effgrad = mom1_new / (T.sqrt(mom2_new) + 1e-10)

        effstep_new = lr_t * effgrad

        # Do update
        w_new = shvar + effstep_new

        # Apply update
        updates[shvar] = w_new
        updates[mom1] = mom1_new
        updates[mom2] = mom2_new

    return updates

#=============================================================================
# Trainer base class
class TrainerBase(HyperBase):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **hyper_params):
        super(TrainerBase, self).__init__()

        self.logger = logging.getLogger("trainer")
        self.dlog = dlog.getLogger("trainer")

        self.step = 0

        self.register_hyper_param("model", default=None, help="")
        self.register_hyper_param("dataset", default=None, help="")
        self.register_hyper_param("termination", default=None, help="")
        self.register_hyper_param("final_monitors", default=[], help="")
        self.register_hyper_param("epoch_monitors", default=[], help="")
        self.register_hyper_param("step_monitors", default=[], help="")
        self.register_hyper_param("first_epoch_step_monitors", default=[], help="")
        self.register_hyper_param("monitor_nth_step", default=1, help="")

        self.shvar = {}
        self.shvar_update_fnc = {}

        self.set_hyper_params(hyper_params)

    def mk_shvar(self, name, init, update_fnc=None):
        if update_fnc is None:
            update_fnc = lambda self: self.get_hyper_param(name)
        value = init
        if isinstance(value, np.ndarray):
            if (value.dtype == np.float32) or (value.dtype == np.float64):
                value = value.astype(floatX)
        elif isinstance(value, float):
            value = np.asarray(value, dtype=floatX)
        elif isinstance(value, int):
            pass
        else:
            raise ArgumentError('Unknown datatype')
        self.shvar[name] = theano.shared(value, name=name, allow_downcast=True)
        self.shvar_update_fnc[name] = update_fnc

    def update_shvars(self):
        for key, shvar in iteritems(self.shvar):
            value = self.shvar_update_fnc[key](self)
            if isinstance(value, np.ndarray):
                if (value.dtype == np.float32) or (value.dtype == np.float64):
                    value = value.astype(floatX)
            shvar.set_value(value)

    def load_data(self):
        dataset = self.dataset
        assert isinstance(dataset, DataSet)

        n_datapoints = dataset.n_datapoints
        assert n_datapoints == dataset.X.shape[0]

        X, Y = dataset.preproc(dataset.X, dataset.Y)
        self.train_X = theano.shared(X, "train_X")
        self.train_Y = theano.shared(Y, "train_Y")

        self.train_perm = theano.shared(np.random.permutation(n_datapoints))

    def shuffle_train_data(self):
        n_datapoints = self.dataset.n_datapoints
        self.train_perm.set_value(np.random.permutation(n_datapoints))

    @abc.abstractmethod
    def compile(self):
        pass


#=============================================================================
# Doubly stochastic gradient Nose-Hoover thermostat (DSGNHT) trainer for DGM (with recognition model)
#=============================================================================
class DSGNHTTrainer(TrainerBase):
    def __init__(self, **hyper_params):
        super(DSGNHTTrainer, self).__init__()

        self.epoch = 1
        self.last_valid_model = None

        self.register_hyper_param("posterior_mean_monitor", default=None, help="help to compute posterior mean")
        self.register_hyper_param("posterior_mean_samples", default=[100], help="# of samples for computing posterior mean")
        self.register_hyper_param("post_optm_steps", default=10, help="# of steps of updating recognition model after get posterior mean")

        self.register_hyper_param("batch_size", default=100, help="size of mini-batch")
        self.register_hyper_param("n_samples", default=5, help="No. samples used during training")
        self.register_hyper_param("n_steps_simu", default=10, help="No. steps used during simulating the dynamics")
        self.register_hyper_param("n_steps_optm", default=1, help="No. steps used during optimizing the recognition model")
        self.register_hyper_param("learning_rate_p", default=0.05, help="Learning rate of generative model")
        self.register_hyper_param("learning_rate_q", default=1e-3, help="Learning rate of recognition model")
        self.register_hyper_param("lr_decay", default=1.0, help="Learning rated decay per epoch")
        self.register_hyper_param("momentum_decay", default=0.05, help="Momentum decay")

        self.mk_shvar('batch_size', 100)
        self.mk_shvar('n_samples', 5)
        self.mk_shvar('eta', 1.0, lambda self: self.learning_rate_p / (self.lr_decay**self.epoch) / self.dataset.n_datapoints)
        self.mk_shvar('nois_var', 1.0, lambda self: 2 * self.learning_rate_p / (self.lr_decay**self.epoch) / self.dataset.n_datapoints * self.momentum_decay)
        self.mk_shvar('momentum_decay', 0.05)

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

        batch_size = self.shvar['batch_size']
        n_samples = self.shvar['n_samples']
        eta = self.shvar['eta']
        nois_var = self.shvar['nois_var']
        momentum_decay = self.shvar['momentum_decay']

        batch_idx = T.iscalar('batch_idx')
        batch_idx.tag.test_value = 0

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch = self.train_X[self.train_perm[first:last]]
        #Y_batch = self.train_Y[self.train_perm[first:last]]

        X_batch, _ = self.dataset.late_preproc(X_batch, None)

        batch_log_PX, grads_p, grads_q = model.get_gradients(
                    X_batch, None,
                    n_samples=n_samples
                )
        log_PX = batch_log_PX / batch_size

        #---------------------------------------------------------------------
        self.logger.info("compiling do_step_simu")
        # Initialize generative model momentum variables
        momentums_p = {}
        alphas = {}
        for shvar, grad in iteritems(grads_p):
            name = shvar.name
            momentums_p[shvar] = theano.shared(np.random.normal(scale=np.sqrt(self.learning_rate_p
                / n_datapoints),size=shvar.get_value().shape).astype(floatX), name=("%s_mom"%name))
            alphas[shvar] = theano.shared(shvar.get_value()*0.+self.momentum_decay, name=("%s_alpha"%name))

        updates_p = OrderedDict()
        for shvar, grad in iteritems(grads_p):
            momentum_p = momentums_p[shvar]
            alpha = alphas[shvar]
            grad_like, grad_prior = grad

            # simulate dynamics here
            new_momentum_p = T.switch(T.isnan(grad_like + grad_prior), # check if nan in grad_like or grad_prior
                momentum_p,
                (1.-alpha)*momentum_p - T.cast(eta * (grad_prior+grad_like/batch_size*n_datapoints), dtype=floatX) + T.sqrt(nois_var)*theano_rng.normal(size=shvar.shape, dtype=floatX)
            )

            new_alpha = alpha + (momentum_p * momentum_p - eta)

            updates_p[momentum_p] = new_momentum_p
            updates_p[shvar] = shvar + new_momentum_p
            # updates_p[shvar] = T.clip(shvar + new_momentum_p,-10,10)
            updates_p[alpha] = new_alpha

        self.do_step_simu = theano.function(
                            inputs=[batch_idx],
                            outputs=log_PX,
                            updates=updates_p,
                            name="do_step_simu")

        #---------------------------------------------------------------------
        self.logger.info("compiling do_step_optm")
        updates_q = get_adam_updates(grads_q, learning_rate=self.learning_rate_q, weight_decay=float(self.batch_size)/n_datapoints)
        self.do_step_optm = theano.function(
                            inputs=[batch_idx],
                            outputs=[], #, Lp, Lq, w],
                            updates=updates_q,
                            name="do_step_optm")


    def perform_learning(self):
        self.update_shvars()

        termination = self.termination
        model = self.model

        assert isinstance(termination, Termination)
        assert isinstance(model, Model)

        # Print information
        n_datapoints = self.dataset.n_datapoints
        n_batches = n_datapoints // self.batch_size

        self.logger.info("Dataset contains %d datapoints in %d mini-batches (%d datapoints per mini-batch)" %
            (n_datapoints, n_batches, self.batch_size))
        self.logger.info("Using %d training samples" % self.n_samples)
        self.logger.info("lr_p=%3.1e, lr_q=%3.1e, lr_decay=%5.1e" %
            (self.learning_rate_p, self.learning_rate_q, self.lr_decay))

        # Perform first epoch
        saved_step_monitors = self.step_monitors
        self.step_monitors = self.first_epoch_step_monitors + self.step_monitors

        for m in self.step_monitors + self.epoch_monitors:
            m.on_init(model)
            m.on_iter(model)

        self.logger.info("Starting epoch 1...")
        #L = self.perform_epoch(only_rc=True)
        L = self.perform_epoch()
        self.step_monitors = saved_step_monitors

        # remaining epochs...
        termination.reset()
        while termination.continue_learning(L):
            self.epoch = self.epoch + 1
            self.logger.info("Starting epoch %d..." % self.epoch)
            L = self.perform_epoch()

        termination_epoch = self.epoch

        post = self.posterior_mean_monitor
        post.on_init(model)

        for m in self.final_monitors:
            m.on_init(model)

        for post_it in range(1, 1 + max(self.posterior_mean_samples)):
            self.logger.info("Collecting posterior samples (%d/%d)..." % (post_it, max(self.posterior_mean_samples)))

            self.epoch = self.epoch + 1
            self.logger.info("Starting epoch %d (terminated at %d) ..." % (self.epoch, termination_epoch))
            L = self.perform_epoch()
            post.on_iter(model)

            if post_it in self.posterior_mean_samples:
                post.average(model)

                for post_optm_it in range(self.post_optm_steps):
                    L = self.perform_epoch(only_rc=True)

                # run final_monitors ...
                self.logger.info("Calling final_monitors...")
                for m in self.final_monitors:
                    m.on_iter(model)

    #-----------------------------------------------------------------------
    def perform_epoch(self, only_rc=False):
        n_datapoints = self.dataset.n_datapoints
        batch_size = self.batch_size
        n_batches = n_datapoints // batch_size
        LL_epoch = 0

        self.update_shvars()
        self.shuffle_train_data()

        if only_rc:
            widget_prefix = "(Model fixed) "
        else:
            widget_prefix = ""

        widgets = [widget_prefix + "Epoch %d, step "%(self.epoch), pbar.Counter(), ' (', pbar.Percentage(), ') ', pbar.Bar(), ' ', pbar.Timer(), ' ', pbar.ETA()]
        bar = pbar.ProgressBar(widgets=widgets, maxval=n_batches).start()

        t0 = time()
        while True:
            LL = self.perform_step(only_rc=only_rc)
            LL_epoch += LL

            batch_idx = self.step % n_batches
            bar.update(batch_idx)

            if self.step % n_batches == 0:
                break
        t = time()-t0
        bar.finish()

        LL_epoch /= n_batches

        if math.isnan(LL_epoch):
            self.model.model_params_from_dict(self.last_valid_model)
            self.logger.error("Epoch %d, found NaN!! Rolled back to last valid model." % self.epoch)
        else:
            self.last_valid_model = self.model.model_params_to_dict()

        if only_rc:
            return LL_epoch

        self.logger.info("Completed epoch %d in %.1fs (%.1fms/step). Calling epoch_monitors..." % (self.epoch, t, t/n_batches*1000))
        for m in self.epoch_monitors:
            m.on_iter(self.model)

        self.dlog.append_all({
            'timing.epoch':  t,
            'timing.step': t/n_batches
        })
        return LL_epoch

    def perform_step(self, only_rc=False):
        n_batches = self.dataset.n_datapoints // self.batch_size
        batch_idx = self.step % n_batches

        LL = 0
        for i in range(self.n_steps_simu):
            for j in range(self.n_steps_optm):
                self.do_step_optm(batch_idx)
            if not only_rc:
                LL = self.do_step_simu(batch_idx)

        self.step = self.step + 1
        batch_idx = self.step % n_batches

        self.dlog.append("pstep_L", LL)

        if (self.step % self.monitor_nth_step == 0) and (len(self.step_monitors) > 0):
            self.logger.info("Epoch %d, step %d (%d steps total): Calling step_monitors...\x1b[K" % (self.epoch, batch_idx, self.step))
            for m in self.step_monitors:
                m.on_iter(self.model)

        return LL

#=============================================================================
# Doubly stochastic gradient Nose-Hoover thermostat (DSGNHT) trainer for DGM (without recognition model)
#=============================================================================
# This method use a Gibbs sampler to sample from p(H | X) instead of using a recognition model
class DSGNHTGibbsTrainer(TrainerBase):
    def __init__(self, **hyper_params):
        super(DSGNHTGibbsTrainer, self).__init__()

        self.epoch = 1
        self.last_valid_model = None

        self.register_hyper_param("posterior_mean_monitor", default=None, help="help to compute posterior mean")
        self.register_hyper_param("posterior_mean_samples", default=[100], help="# of samples for computing posterior mean")

        self.register_hyper_param("batch_size", default=100, help="size of mini-batch")
        self.register_hyper_param("n_samples", default=5, help="No. samples used during training")
        self.register_hyper_param("n_steps_simu", default=10, help="No. steps used during simulating the dynamics")
        self.register_hyper_param("learning_rate_p", default=0.1, help="Learning rate of generative model")
        self.register_hyper_param("lr_decay", default=1.0, help="Learning rated decay per epoch")
        self.register_hyper_param("momentum_decay", default=0.1, help="Momentum decay")

        self.mk_shvar('batch_size', 100)
        self.mk_shvar('n_samples', 5)
        self.mk_shvar('eta', 1.0, lambda self: self.learning_rate_p / (self.lr_decay**self.epoch) / self.dataset.n_datapoints)
        self.mk_shvar('nois_var', 1.0, lambda self: 2 * self.learning_rate_p / (self.lr_decay**self.epoch) / self.dataset.n_datapoints * self.momentum_decay)
        self.mk_shvar('momentum_decay', 0.1)

        self.set_hyper_params(hyper_params)
        self.logger.debug("hyper_params %s" % self.get_hyper_params())

    def load_data(self):
        super(DSGNHTGibbsTrainer, self).load_data()

        self.logger.info("called overrided load_data()")

        n_datapoints = self.dataset.n_datapoints
        sum_hidden_dim, _ = self.model.get_hidden_dim()

        H = np.zeros((n_datapoints * self.n_samples, sum_hidden_dim), dtype=floatX)
        self.train_H = theano.shared(H, "train_H")

    def compile(self):
        """ Theano-compile neccessary functions """
        from learning.utils.misc import f_getid_from_replicated

        model = self.model

        assert isinstance(model, Model)

        model.setup()
        self.update_shvars()
        #---------------------------------------------------------------------
        self.logger.info("compiling ...")

        n_datapoints = self.dataset.n_datapoints

        batch_size = self.shvar['batch_size']
        n_samples = self.shvar['n_samples']
        eta = self.shvar['eta']
        nois_var = self.shvar['nois_var']
        momentum_decay = self.shvar['momentum_decay']

        batch_idx = T.iscalar('batch_idx')
        batch_idx.tag.test_value = 0

        first = batch_idx*batch_size
        last  = first + batch_size
        X_batch = self.train_X[self.train_perm[first:last]]
        Hid = f_getid_from_replicated(self.train_perm[first:last], n_samples)
        H_batch = self.train_H[Hid]

        X_batch, _ = self.dataset.late_preproc(X_batch, None)

        batch_log_PX, grads_p, new_H_batch = model.get_gradients(
                    X_batch, H_batch,
                    n_samples=n_samples
                )
        log_PX = batch_log_PX / batch_size

        #---------------------------------------------------------------------
        self.logger.info("compiling do_step_simu")
        # Initialize generative model momentum variables
        momentums_p = {}
        alphas = {}
        for shvar, grad in iteritems(grads_p):
            name = shvar.name
            momentums_p[shvar] = theano.shared(np.random.normal(scale=np.sqrt(self.learning_rate_p
                / n_datapoints),size=shvar.get_value().shape).astype(floatX), name=("%s_mom"%name))
            alphas[shvar] = theano.shared(shvar.get_value()*0.+self.momentum_decay, name=("%s_alpha"%name))

        updates_p = OrderedDict()
        for shvar, grad in iteritems(grads_p):
            momentum_p = momentums_p[shvar]
            alpha = alphas[shvar]
            grad_like, grad_prior = grad

            # simulate dynamics here
            new_momentum_p = T.switch(T.isnan(grad_like + grad_prior), # check if nan in grad_like or grad_prior
                momentum_p,
                (1.-alpha)*momentum_p - T.cast(eta * (grad_prior+grad_like/batch_size*n_datapoints), dtype=floatX) + T.sqrt(nois_var)*theano_rng.normal(size=shvar.shape, dtype=floatX)
            )

            new_alpha = alpha + (momentum_p * momentum_p - eta)

            updates_p[momentum_p] = new_momentum_p
            updates_p[shvar] = shvar + new_momentum_p
            # updates_p[shvar] = T.clip(shvar + new_momentum_p,-10,10)
            updates_p[alpha] = new_alpha
            updates_p[self.train_H] = T.set_subtensor(H_batch, new_H_batch)

        self.do_step_simu = theano.function(
                            inputs=[batch_idx],
                            outputs=log_PX,
                            updates=updates_p,
                            name="do_step_simu")

        #---------------------------------------------------------------------
        self.logger.info("compiling do_init_hidden")

        H_init = model.init_hidden(self.train_X, n_samples)
        updates_h = OrderedDict()
        updates_h[self.train_H] = H_init
        self.do_init_hidden = theano.function(inputs=[], outputs=[], updates=updates_h,
                name="do_init_hidden")

    def perform_learning(self):
        self.update_shvars()

        termination = self.termination
        model = self.model

        assert isinstance(termination, Termination)
        assert isinstance(model, Model)

        # Print information
        n_datapoints = self.dataset.n_datapoints
        n_batches = n_datapoints // self.batch_size

        self.logger.info("Dataset contains %d datapoints in %d mini-batches (%d datapoints per mini-batch)" %
            (n_datapoints, n_batches, self.batch_size))
        self.logger.info("Using %d training samples" % self.n_samples)
        self.logger.info("lr_p=%3.1e, lr_decay=%5.1e" %
            (self.learning_rate_p, self.lr_decay))

        # initialize hidden variables
        self.do_init_hidden()

        # Perform first epoch
        saved_step_monitors = self.step_monitors
        self.step_monitors = self.first_epoch_step_monitors + self.step_monitors

        for m in self.step_monitors + self.epoch_monitors:
            m.on_init(model)
            m.on_iter(model)

        self.logger.info("Starting epoch 1...")
        L = self.perform_epoch()
        self.step_monitors = saved_step_monitors

        # remaining epochs...
        termination.reset()
        while termination.continue_learning(L):
            self.epoch = self.epoch + 1
            self.logger.info("Starting epoch %d..." % self.epoch)
            L = self.perform_epoch()

        termination_epoch = self.epoch

        post = self.posterior_mean_monitor
        post.on_init(model)

        for m in self.final_monitors:
            m.on_init(model)

        for post_it in range(1, 1 + max(self.posterior_mean_samples)):
            self.logger.info("Collecting posterior samples (%d/%d)..." % (post_it, max(self.posterior_mean_samples)))

            self.epoch = self.epoch + 1
            self.logger.info("Starting epoch %d (terminated at %d) ..." % (self.epoch, termination_epoch))
            L = self.perform_epoch()
            post.on_iter(model)

            if post_it in self.posterior_mean_samples:
                post.average(model)

                # run final_monitors ...
                self.logger.info("Calling final_monitors...")
                for m in self.final_monitors:
                    m.on_iter(model)

    #-----------------------------------------------------------------------
    def perform_epoch(self):
        n_datapoints = self.dataset.n_datapoints
        batch_size = self.batch_size
        n_batches = n_datapoints // batch_size
        LL_epoch = 0

        self.update_shvars()
        self.shuffle_train_data()

        widgets = ["Epoch %d, step "%(self.epoch), pbar.Counter(), ' (', pbar.Percentage(), ') ', pbar.Bar(), ' ', pbar.Timer(), ' ', pbar.ETA()]
        bar = pbar.ProgressBar(widgets=widgets, maxval=n_batches).start()

        t0 = time()
        while True:
            LL = self.perform_step()
            LL_epoch += LL

            batch_idx = self.step % n_batches
            bar.update(batch_idx)

            if self.step % n_batches == 0:
                break
        t = time()-t0
        bar.finish()

        LL_epoch /= n_batches

        if math.isnan(LL_epoch):
            self.model.model_params_from_dict(self.last_valid_model)
            self.logger.error("Epoch %d, found NaN!! Rolled back to last valid model." % self.epoch)
        else:
            self.last_valid_model = self.model.model_params_to_dict()

        self.logger.info("Completed epoch %d in %.1fs (%.1fms/step). Calling epoch_monitors..." % (self.epoch, t, t/n_batches*1000))
        for m in self.epoch_monitors:
            m.on_iter(self.model)

        self.dlog.append_all({
            'timing.epoch':  t,
            'timing.step': t/n_batches
        })
        return LL_epoch

    def perform_step(self):
        n_batches = self.dataset.n_datapoints // self.batch_size
        batch_idx = self.step % n_batches

        LL = 0
        for i in range(self.n_steps_simu):
            LL = self.do_step_simu(batch_idx)

        self.step = self.step + 1
        batch_idx = self.step % n_batches

        self.dlog.append("pstep_L", LL)

        if (self.step % self.monitor_nth_step == 0) and (len(self.step_monitors) > 0):
            self.logger.info("Epoch %d, step %d (%d steps total): Calling step_monitors...\x1b[K" % (self.epoch, batch_idx, self.step))
            for m in self.step_monitors:
                m.on_iter(self.model)

        return LL
