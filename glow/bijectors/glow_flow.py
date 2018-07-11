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
                 num_levels=2,
                 level_depth=2,
                 event_ndims=3,
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
        assert event_ndims == 3, event_ndims
        assert event_dims is not None and len(event_dims) == 3, event_dims

        self._graph_parents = []
        self._name = name
        self._validate_args = validate_args

        self._num_levels = num_levels
        self._level_depth = level_depth

        self._event_dims = event_dims

        self.build()

        super().__init__(event_ndims=event_ndims,
                         validate_args=validate_args,
                         name=name)

    def build(self):
        flow_parts = []

        for l in range(self._num_levels):
            out = squeeze(out)
            level_flow_parts = []

            for k in range(self._level_depth):
                shape = out.get_shape()
                image_shape = shape[1:]

                activation_normalization = tfb.BatchNormalization(
                    batchnorm_layer=tf.layers.BatchNormalization(axis=-1))
                convolution_permute = ConvolutionPermute(
                    event_ndims=3, event_dims=shape[1:])
                flatten = tfb.Reshape(event_shape_out=(-1, np.prod(image_shape)))
                affine_coupling = tfb.RealNVP(
                    num_masked=np.prod(image_shape)//2,
                    shift_and_log_scale_fn=glow_resnet_template(
                        hidden_layers=hidden_sizes,
                        activation=tf.nn.relu))
                unflatten = tfb.Reshape(event_shape_out=shape)

                level_flow = tfb.Chain(reversed([
                    activation_normalization,
                    convolution_permute,
                    flatten,
                    affine_coupling,
                    unflatten,
                ]))

                level_flow_parts.append(level_flow)

            flow_parts.append(level_flow_parts)

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


def glow_resnet_template(
        hidden_layers,
        shift_only=False,
        activation=tf.nn.relu,
        name=None,
        *args,
        **kwargs):
  """Build a scale-and-shift functions using a weight normalized resnet.
  This will be wrapped in a make_template to ensure the variables are only
  created once. It takes the `d`-dimensional input x[0:d] and returns the `D-d`
  dimensional outputs `loc` ("mu") and `log_scale` ("alpha").
  Arguments:
    TODO
  Returns:
    shift: `Float`-like `Tensor` of shift terms.
    log_scale: `Float`-like `Tensor` of log(scale) terms.
  Raises:
    NotImplementedError: if rightmost dimension of `inputs` is unknown prior to
      graph execution.
  #### References
  TODO
  """

  with ops.name_scope(name, "glow_resnet_template"):
    def _fn(x, output_units):
      """Weight normalized resnet parameterized via `glow_resnet_template`."""

      # TODO: implement resnet
      x = None

      if shift_only:
        return x, None
      shift, log_scale = array_ops.split(x, 2, axis=-1)
      return shift, log_scale
    return template_ops.make_template(
        "glow_resnet_template", _fn)
