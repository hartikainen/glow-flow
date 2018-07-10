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
                 event_ndims=1,
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
        assert event_ndims == 1, event_ndims
        assert event_dims is not None and len(event_dims) == 1, event_dims

        self._graph_parents = []
        self._name = name
        self._validate_args = validate_args

        self._event_dims = event_dims

        super().__init__(event_ndims=event_ndims,
                         validate_args=validate_args,
                         name=name)

    def _forward(self, x):
        raise NotImplementedError('_forward')

    def _inverse(self, y):
        raise NotImplementedError('_inverse')

    def _forward_log_det_jacobian(self, x):
        raise NotImplementedError('_forward_log_det_jacobian')

    def _inverse_log_det_jacobian(self, y):
        raise NotImplementedError('_inverse_log_det_jacobian')

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
