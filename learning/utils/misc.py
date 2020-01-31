import numpy as np

import theano
import theano.tensor as T

from learning.setting import *

#=============================================================================

def enumerate_pairs(start, end):
    return [(i, i+1) for i in xrange(0, end-1)]

def f_replicate_batch(A, repeat):
    """Extend the given 2d Tensor by repeating reach line *repeat* times.

    With A.shape == (rows, cols), this function will return an array with
    shape (rows*repeat, cols).

    Parameters
    ----------
    A : T.tensor
        Each row of this 2d-Tensor will be replicated *repeat* times
    repeat : int

    Returns
    -------
    B : T.tensor
    """
    A_ = A.dimshuffle((0, 'x', 1))
    A_ = A_ + T.zeros((A.shape[0], repeat, A.shape[1]), dtype=floatX)
    A_ = A_.reshape( [A_.shape[0]*repeat, A.shape[1]] )
    return A_

def f_getid_from_replicated(ids, n_samples):
    return T.repeat(ids * n_samples, n_samples) + T.tile(T.arange(n_samples), ids.shape[0])

def f_logsumexp(A, axis=None):
    """Numerically stable log( sum( exp(A) ) ) """
    A_max = T.max(A, axis=axis, keepdims=True)
    B = T.log(T.sum(T.exp(A-A_max), axis=axis, keepdims=True))+A_max
    B = T.sum(B, axis=axis)
    return B

