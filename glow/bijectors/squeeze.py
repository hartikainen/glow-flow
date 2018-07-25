"""Squeeze bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp

from .transpose import Transpose


__all__ = [
    "Squeeze",
]


tfb = tfp.bijectors


class Squeeze(tfb.Bijector):
    """Squeeze Bijector

    TODO: Should probably be renamed since this does not only squeeze,
    but also transposes some axis in between.
    """

    def __init__(self,
                 factor=2,
                 forward_min_event_ndims=3,
                 inverse_min_event_ndims=3,
                 *args,
                 **kwargs):
        """Instantiates the `Squeeze` reshaping bijector."""
        self._factor = factor
        self.built = False

        # Note that the super is not for Squeeze but Reshape
        super(Squeeze, self).__init__(
            forward_min_event_ndims=forward_min_event_ndims,
            inverse_min_event_ndims=inverse_min_event_ndims,
            *args,
            **kwargs)

    def build(self, forward_input_shape=None, inverse_input_shape=None):
        factor = self._factor

        if forward_input_shape is not None:
            assert inverse_input_shape is None, inverse_input_shape
            (H, W, C) = input_shape = forward_input_shape[-3:]
            intermediate_shape = (H//factor, factor, W//factor, factor, C)
            output_shape = (H//factor, W//factor, C*factor**2)
        elif inverse_input_shape is not None:
            assert forward_input_shape is None, forward_input_shape
            (H, W, C) = output_shape = inverse_input_shape[-3:]
            intermediate_shape = (H, W, C//factor**2, factor, factor)
            input_shape = (H*factor, W*factor, C//factor**2)
        else:
            raise ValueError(
                "Exactly one of {forward_input_shape, inverse_input_shape}"
                " must be defined.")

        self._bijectors = [
            tfb.Reshape(
                event_shape_in=input_shape,
                event_shape_out=intermediate_shape),
            Transpose(perm=[0, 1, 3, 5, 2, 4]),
            tfb.Reshape(
                event_shape_in=intermediate_shape,
                event_shape_out=output_shape),
        ]
        self._flow = tfb.Chain(list(reversed(self._bijectors)))
        self.built = True

    def _forward(self, x):
        if not self.built:
            self.build(forward_input_shape=x.shape.as_list())
        return self._flow.forward(x)

    def _inverse(self, y):
        if not self.built:
            self.build(inverse_input_shape=y.shape.as_list())
        return self._flow.inverse(y)

    def _forward_log_det_jacobian(self, x, *args, **kwargs):
        return tf.constant(0.0, dtype=x.dtype)

    def _inverse_log_det_jacobian(self, y, *args, **kwargs):
        return tf.constant(0.0, dtype=y.dtype)
