"""Squeeze bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_probability as tfp


__all__ = [
    "Squeeze",
]


tfb = tfp.bijectors


class Squeeze(tfb.Reshape):
    """TODO"""

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
        super(tfb.Reshape, self).__init__(
            forward_min_event_ndims=forward_min_event_ndims,
            inverse_min_event_ndims=inverse_min_event_ndims,
            *args,
            **kwargs)

    def build(self, forward_input_shape=None, inverse_input_shape=None):
        factor = self._factor

        if forward_input_shape is not None:
            assert inverse_input_shape is None, inverse_input_shape
            (H, W, C) = input_shape = forward_input_shape[-3:]
            output_shape = (H//factor, W//factor, C*factor*factor)
        elif inverse_input_shape is not None:
            assert forward_input_shape is None, forward_input_shape
            (H, W, C) = output_shape = inverse_input_shape[-3:]
            input_shape = (H*factor, W*factor, C/(factor*factor))
        else:
            raise ValueError(
                "Exactly one of {forward_input_shape, inverse_input_shape}"
                " must be defined.")

        self.built = True

        tfb.Reshape.__init__(
            self,
            event_shape_out=output_shape,
            event_shape_in=input_shape,
            validate_args=self._validate_args,
            name=self._name)

    def _forward(self, x):
        if not self.built:
            self.build(forward_input_shape=x.shape.as_list())
        y = tfb.Reshape._forward(self, x)
        return y

    def _inverse(self, y):
        if not self.built:
            self.build(inverse_input_shape=y.shape.as_list())
        x = tfb.Reshape._inverse(self, y)
        return x

    def _forward_log_det_jacobian(self, x):
        if not self.built:
            self.build(forward_input_shape=x.shape.as_list())

        return tfb.Reshape._forward_log_det_jacobian(self, x)

    def _inverse_log_det_jacobian(self, y):
        if not self.built:
            self.build(inverse_input_shape=y.shape.as_list())

        return tfb.Reshape._inverse_log_det_jacobian(self, y)
