#!/usr/bin/env python


from __future__ import division

import os
import sys
sys.path.append("../")

import logging
import coloredlogs
from time import time

import numpy as np

logger = logging.getLogger()


def run_experiment(args):
    # set environment variable for theano
    os.environ['THEANO_FLAGS'] = "device=gpu" + str(args.gpu)

    from learning.experiment import Experiment
    from learning.training import DSGNHTTrainer
    from learning.termination import EarlyStopping
    from learning.monitor import MonitorLL, DLogModelParams, SampleFromP, MonitorPosteriorMean
    from learning.dataset import CalTech101Silhouettes
    from learning.preproc import PermuteColumns, Binarize

    from learning.models.dgm  import DGMLayerStack
    from learning.models.sbn  import SBN, SBNTop
    from learning.models.dsbn import DSBN
    from learning.models.darn import DARN, DARNTop
    from learning.models.nade import NADE, NADETop

    np.set_printoptions(precision=2)

    logger.debug("Arguments %s" % args)
    tags = []

    np.random.seed(23)

    # Layer models
    layer_models = {
        "sbn" : (SBN, SBNTop),
        "dsbn": (DSBN, SBNTop),
        "darn": (DARN, DARNTop),
        "nade": (NADE, NADETop),
    }

    assert args.p_model in layer_models
    assert args.q_model in layer_models

    p_layer, p_top = layer_models[args.p_model]
    q_layer, q_top = layer_models[args.q_model]

    # n_samples to evaluate model
    n_samples_epoch = [1, 5, 25, 100]
    n_samples_final = [1, 5, 10, 25, 100, 500, 1000, 10000, 100000]
    if (args.p_model in ['darn', 'nade']) or (args.q_model in ['darn', 'nade']):
        n_samples_epoch = [1, 5, 25]
        n_samples_final = [1, 5, 10, 25, 100, 500]

    # no. posterior samples for posterior mean
    postsamples = [int(s) for s in args.postsamples.split(",")]

    # Layer sizes
    layer_sizes = [int(s) for s in args.layer_sizes.split(",")]

    n_X = 28*28

    p_layers = []
    q_layers = []


    for ls in layer_sizes:
        n_Y = ls
        p_layers.append(
            p_layer(n_X=n_X, n_Y=n_Y)
        )
        q_layers.append(
            q_layer(n_X=n_Y, n_Y=n_X)
        )
        n_X = n_Y
    p_layers.append( p_top(n_X=n_X) )

    model = DGMLayerStack(
        p_layers=p_layers,
        q_layers=q_layers
    )
    model.setup()

    # parameters
    def param_tag(value):
        """ Convert a float into a short tag-usable string representation. E.g.:
            0.1   -> 11
            0.01  -> 12
            0.001 -> 13
            0.005 -> 53
        """
        if value == 0.0:
            return "00"
        exp = np.floor(np.log10(value))
        leading = ("%e"%value)[0]
        return "%s%d" % (leading, -exp)

    # Learning rate
    lr_p = args.lr_p
    tags += ["lp"+param_tag(lr_p)]
    lr_q = args.lr_q
    tags += ["lq"+param_tag(lr_q)]

    # LR decay
    if args.lrdecay != 1.0:
        tags += ["lrdecay"+param_tag(args.lrdecay-1.)]

    # Samples
    n_samples = args.samples
    tags += ["spl%d"%n_samples]

    # Batch size
    batch_size = args.batchsize
    tags += ["bs%d"%batch_size]

    # n_steps_simu
    n_steps_simu = args.n_simu
    tags += ["ns%d"%n_steps_simu]

    # n_steps_optm
    n_steps_optm = args.n_optm
    tags += ["no%d"%n_steps_optm]

    # momentum_decay
    momentum_decay = args.momentum_decay
    tags += ["md"+param_tag(momentum_decay)]

    # Dataset
    if args.shuffle:
        np.random.seed(23)
        preproc = [PermuteColumns()]
        tags += ["shuffle"]
    else:
        preproc = []

    dataset = CalTech101Silhouettes(which_set='train', preproc=preproc)
    valiset = CalTech101Silhouettes(which_set='valid', preproc=preproc)
    testset = CalTech101Silhouettes(which_set='test', preproc=preproc)

    # lookahead
    lookahead = args.lookahead
    tags += ["lah%d" % lookahead]

    tags.sort()
    expname = "dsgnht-%s-%s-%s-%s"% ("-".join(tags), args.p_model, args.q_model, "-".join([str(s) for s in layer_sizes]))

    if args.report:
        expname = "report/" + expname

    logger.info("Running %s" % expname)


#-----------------------------------------------------------------------------

    dlog_model_params_monitor = DLogModelParams()
    generate_data_monitor = SampleFromP(n_samples=100)

    trainer = DSGNHTTrainer(
        batch_size=batch_size,
        n_samples=n_samples,
        n_steps_simu=n_steps_simu,
        n_steps_optm=n_steps_optm,
        learning_rate_p=lr_p,
        learning_rate_q=lr_q,
        lr_decay=args.lrdecay,
        momentum_decay=momentum_decay,
        dataset=dataset,
        model=model,
        termination=EarlyStopping(lookahead=lookahead, min_epochs=10, max_epochs=999999),
        epoch_monitors=[
            dlog_model_params_monitor,
            generate_data_monitor,
            MonitorLL(name="valiset", data=valiset, n_samples=n_samples_epoch),
        ],
        final_monitors=[
            dlog_model_params_monitor,
            generate_data_monitor,
            MonitorLL(name="final-testset", data=testset, n_samples=n_samples_final,
                level=logging.CRITICAL),
        ],
        posterior_mean_samples=postsamples,
        posterior_mean_monitor=MonitorPosteriorMean(),
    )

    experiment = Experiment()
    experiment.set_trainer(trainer)
    experiment.setup_output_dir(expname)
    experiment.setup_logging()
    experiment.print_summary()

    if args.cont is None:
        logger.info("Starting experiment ...")
        experiment.run_experiment()
    else:
        logger.info("Continuing experiment %s ...." % args.cont)
        experiment.continue_experiment(args.cont+"/results.h5", row=-1)

    logger.info("Finished. Wrinting metadata")

    experiment.print_summary()

#=============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--report', default=False, action="store_true",
        help="Store results in report folder. (default: False)")

    parser.add_argument('--shuffle', default=False, action='store_true',
        help="Shuffle the data. (default: False)")

    parser.add_argument('--cont', nargs='?', default=None,
        help="Continue a previous in result_dir")

    parser.add_argument('--gpu', default=0, type=int,
        help="ID of gpu device to use. (default: 0)")

    parser.add_argument('--samples', default=5, type=int,
        help="Number of training samples. (default: 5)")

    parser.add_argument('--batchsize', default=100, type=int,
        help="Mini batch size. (default: 100)")

    parser.add_argument('--n_simu', default=10, type=int,
        help="No. steps used during simulating the dynamics. (default: 10)")

    parser.add_argument('--n_optm', default=1, type=int,
        help="No. steps used during optimizing the recognition model. (default: 1)")

    parser.add_argument('--lr_p', default=0.005, type=float,
        help="Per-batch learning rate of DSGNHT. (default: 0.005)")

    parser.add_argument('--lr_q', default=0.0003, type=float,
        help="Step size of Adam. (default: 0.0003)")

    parser.add_argument('--lrdecay', default=1., type=float,
        help="Learning rate decay. (default: 1.0)")

    parser.add_argument('--momentum_decay', default=0.1, type=float,
        help="Momentum decay of DSGNHT. (default: 0.1)")

    parser.add_argument('--lookahead', default=10, type=int,
        help="Termination criteria: # epochs without LL increase. (default: 10)")

    parser.add_argument('--postsamples', default="1,5,10,20,50,100,200", type=str,
        help="# samples for computing posterior mean. (default: 1,5,10,20,50,100,200)")

    parser.add_argument('p_model', default="SBN",
        help="SBN, DARN or NADE (Required)")
    parser.add_argument('q_model', default="SBN",
        help="SBN, DARN or NADE (Required)")

    parser.add_argument('layer_sizes', default="200",
        help="Comma seperated list of sizes. Layer cosest to the data comes first. (Required)")

    args = parser.parse_args()

    FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
    DATEFMT = '%H:%M:%S'
    LEVEL_STYLES = dict(
        debug=dict(color='green'),
        info=dict(),
        verbose=dict(color='blue'),
        warning=dict(color='yellow'),
        error=dict(color='red'),
        critical=dict(color='magenta'))

    coloredlogs.install(level='INFO',fmt=FORMAT, datefmt=DATEFMT, level_styles=LEVEL_STYLES)

    run_experiment(args)

