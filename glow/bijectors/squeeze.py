"""Squeeze bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


__all__ = [
    "Squeeze",
]


tfd = tf.contrib.distributions
tfb = tfd.bijectors


class Squeeze(tfb.Bijector):
    """TODO"""

    def __init__(self, factor=2, *args, **kwargs):
        """Instantiates the `Squeeze` reshaping bijector."""
        self._factor = factor
        super(Squeeze, self).__init__(*args, **kwargs)

    def _forward(self, x):
        raise NotImplementedError('_forward')

    def _inverse(self, y):
        raise NotImplementedError('_inverse')

    def _forward_log_det_jacobian(self, x):
        raise NotImplementedError('_forward_log_det_jacobian')

    def _inverse_log_det_jacobian(self, y):
        raise NotImplementedError('_inverse_log_det_jacobian')

