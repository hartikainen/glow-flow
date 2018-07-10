"""GlowFlow bijector flow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


__all__ = [
    "GlowFlow",
]


tfd = tf.contrib.distributions
tfb = tfd.bijectors


class GlowFlow(tfb.Bijector):
    """TODO"""

    def __init__(self,
                 num_layers=2,
                 event_ndims=1,
                 event_dims=None,
                 validate_args=False,
                 name="glow_flow"):
        """Instantiates the `GlowFlow` normalizing flow.

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

        self._num_layers = num_layers

        self._event_dims = event_dims

        self.build()

        super().__init__(event_ndims=event_ndims,
                         validate_args=validate_args,
                         name=name)

    def build(self):
        num_layers = self._num_layers

        flow_parts = []
        for i in range(num_layers):
            hidden_sizes = (25, 25)
            D = np.prod(self._event_dims)

            coupling_layer = tfb.RealNVP(
                num_masked=D//2,
                shift_and_log_scale_fn=tfb.real_nvp_default_template(
                    hidden_layers=hidden_sizes,
                    # TODO: test tf.nn.relu
                    activation=tf.nn.tanh))

            flow_parts.append(coupling_layer)

            if i < num_layers - 1:
                # TODO: Replace this with 1x1 convolution permutation
                permute_bijector = tfb.Permute(
                    permutation=list(reversed(range(D))))
                flow_parts.append(permute_bijector)

        # Note: tfb.Chain applies the list of bijectors in the _reverse_ order
        # of what they are inputted.
        self.flow = tfb.Chain(list(reversed(flow_parts)))

    def _forward(self, x):
        return self.flow.forward(x)

    def _inverse(self, y):
        return self.flow.inverse(y)

    def _forward_log_det_jacobian(self, x):
        return self.flow.forward_log_det_jacobian(x)

    def _inverse_log_det_jacobian(self, y):
        return  self.flow.inverse_log_det_jacobian(input_)

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
