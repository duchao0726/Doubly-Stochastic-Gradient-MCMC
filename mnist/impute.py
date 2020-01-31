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
    from learning.reconstruct import Reconstructor
    from learning.dataset import MNIST
    from learning.preproc import PermuteColumns, Binarize
    from learning.monitor import MonitorLL, DLogModelParams, SampleFromP, MonitorPosteriorMean

    from learning.models.dgm  import DGMLayerStack
    from learning.models.sbn  import SBN, SBNTop
    from learning.models.dsbn import DSBN
    from learning.models.darn import DARN, DARNTop
    from learning.models.nade import NADE, NADETop

    np.set_printoptions(precision=2)

    logger.debug("Arguments %s" % args)

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


    preproc = []
    # binarize_preproc = preproc + [Binarize(late=True)]
    # dataset = MNIST(which_set='test', preproc=binarize_preproc, n_datapoints=10000)
    dataset = MNIST(fname="mnist_salakhutdinov.pkl.gz", which_set='test', preproc=preproc, n_datapoints=10000)

    expname = "impute/%s-%s-%s"% (args.p_model, args.q_model, "-".join([str(s) for s in layer_sizes]))

    logger.info("Running %s" % expname)


#-----------------------------------------------------------------------------

    trainer = Reconstructor(
        dataset=dataset,
        model=model,
        epoch_monitors=[
            DLogModelParams()
        ],
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
        logger.info("Continuing experiment %s ..." % args.cont)
        experiment.continue_experiment(args.cont+"/results.h5", row=-1)

    logger.info("Finished. Wrinting metadata")

    experiment.print_summary()

#=============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--cont', nargs='?', default=None,
        help="Continue a previous in result_dir")

    parser.add_argument('--gpu', default=0, type=int,
        help="ID of gpu device to use. (default: 0)")

    # model architecture should be detected automatically
    # TO BE IMPLEMENTED
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

