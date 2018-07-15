"""Parallel bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


__all__ = [
    "Parallel",
]


tfd = tf.contrib.distributions
tfb = tfd.bijectors


class Parallel(tfb.Bijector):
    """TODO"""

    def __init__(self,
                 bijectors=None,
                 split_axis=None,
                 split_proportions=None,
                 validate_args=False,
                 name=None):
        """Instantiates `Parallel` bijector.

        Args:
            bijectors: Python `list` of bijector instances. An empty list makes this
                bijector equivalent to the `Identity` bijector.
            validate_args: Python `bool` indicating whether arguments should be
                checked for correctness.
            name: Python `str`, name given to ops managed by this object.
                Default: E.g.,
                `Parallel([Exp(), Softplus()]).name == "parallel_of_exp_and_softplus"`.

        Raises:
            ValueError: if bijectors have different dtypes.
        """

        raise NotImplementedError('__init__')

    def _forward(self, x):
        raise NotImplementedError('_forward')

    def _inverse(self, y):
        raise NotImplementedError('_inverse')

    def _forward_log_det_jacobian(self, x):
        raise NotImplementedError('_forward_log_det_jacobian')

    def _inverse_log_det_jacobian(self, y):
        raise NotImplementedError('_inverse_log_det_jacobian')
