"""Convolution permutation bijectors.

ConvolutionPermute is a (learnable) permutation of image channels.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


__all__ = [
    "ConvolutionPermute",
]


tfd = tf.contrib.distributions
tfb = tfd.bijectors


class ConvolutionPermute(tfb.Bijector):
    """TODO"""

    def __init__(self,
                 event_ndims=3,
                 event_dims=None,
                 validate_args=False,
                 name="convolution_permute"):
        """Instantiates the `ConvolutionPermute` normalizing flow.

        Args:
            TODO
            event_ndims: Python scalar indicating the number of dimensions
                associated with a particular draw from the distribution.
            event_dims: Python list indicating the size of each dimension
                associated with a particular draw from the distribution.
            validate_args: Python `bool` indicating whether arguments should be
                checked for correctness.
            name: Python `str` name given to ops managed by this object.

        Raises:
            ValueError: if TODO happens
        """
        assert event_ndims == 3, event_ndims
        assert event_dims is not None and len(event_dims) == 3, event_dims

        self._graph_parents = []
        self._name = name
        self._validate_args = validate_args

        self._event_dims = event_dims

        W, H, C = event_dims

        self.w = tf.get_variable(
            "w",
            shape=(1, 1, C, C),
            dtype=tf.float32,
            initializer=tf.initializers.orthogonal())

        super().__init__(event_ndims=event_ndims,
                         validate_args=validate_args,
                         name=name)

    def _forward(self, x):
        z = tf.nn.conv2d(
            input=x,
            filter=self.w,
            strides=(1, 1, 1, 1),
            padding='SAME',
            data_format='NHWC')

        return z

    def _inverse(self, y):
        w_inverse = tf.matrix_inverse(self.w)
        x = tf.nn.conv2d(
            input=y,
            filter=w_inverse,
            strides=(1, 1, 1, 1),
            padding='SAME',
            data_format='NHWC')

        return x

    def _forward_log_det_jacobian(self, x):
        H, W, _ = self._event_dims
        determinant = tf.matrix_determinant(tf.cast(self.w, tf.float64))
        log_det_jacobian = (
            H * W * tf.cast(tf.log(abs(determinant)), tf.float32))
        return log_det_jacobian

    def _inverse_log_det_jacobian(self, y):
        return -self.forward_log_det_jacobian(y)

    def _maybe_assert_valid_x(self, x):
        """TODO"""
        if not self.validate_args:
            return x
        raise NotImplementedError("_maybe_assert_valid_x")

    def _maybe_assert_valid_y(self, y):
        """TODO"""
        if not self.validate_args:
            return y
        raise NotImplementedError("_maybe_assert_valid_y")
