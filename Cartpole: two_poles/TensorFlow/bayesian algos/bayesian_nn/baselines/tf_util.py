import numpy as np
import tensorflow as tf
import builtins
import functools
import copy
import os
import numbers


def bayes_dropout(x, keep_prob, noise_shape=None):
    """
    Generates a single dropout mask and applies it.
    """
    keep_prob_orig = keep_prob
    x = tf.convert_to_tensor(x,
                             dtype=tf.float32,
                             name="x")
    if not x.dtype.is_floating:
        raise ValueError("x has to be a floating point tensor since it's going to"
                       " be scaled. Got a %s tensor instead." % x.dtype)
    if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
        raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    keep_prob = tf.convert_to_tensor(keep_prob,
                                     dtype=tf.float32,
                                     name="keep_prob")
        
    # Do nothing if we know keep_prob == 1
    if keep_prob_orig == 1:
        return x
        
    noise_shape = noise_shape if noise_shape is not None else tf.shape(x)
    # uniform [keep_prob, 1.0 + keep_prob)
    np_random_mask = np.floor( keep_prob_orig + np.random.uniform( size=tuple( x.get_shape().as_list()[1:] ) ) )
    binary_tensor = tf.Variable(np_random_mask, trainable=False, dtype=tf.float32)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    ret = tf.math.divide(x, keep_prob) * binary_tensor
    ret.set_shape(x.get_shape())
        
    return ret, binary_tensor