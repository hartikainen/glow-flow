"""Parallel bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

import tensorflow as tf
import numpy as np


__all__ = [
    "Parallel",
]


tfd = tf.contrib.distributions
tfb = tfd.bijectors


def _use_static_shape(input_tensor, ndims):
  return input_tensor.shape.is_fully_defined() and isinstance(ndims, int)


class Parallel(tfb.Bijector):
    """Bijector which applies a set of bijectors in parallel.

    Example Use:

    ```python
    parallel = Parallel([Exp(), Identity()], name="exp_identity")
    ```

    Results in:

    * Forward:

     ```python
     axis = y
     exp = Exp()
     identity = Identity()
     Parallel([exp, identity], axis=axis).forward(x)
     = tf.concatenate([
           exp.forward(tf.split(x, axis=axis)),
           tf.split(x, axis=axis)
       ], axis=axis)
     = tf.concatenate([
           tf.exp(tf.split(x, axis=axis)),
           tf.split(x, axis=axis)
       ], axis=axis)
     ```

    * Inverse:

     ```python
     axis = y
     exp = Exp()
     identity = Identity()
     Parallel([exp, identity], axis=axis).forward(y)
     = tf.concatenate([
           exp.inverse(tf.split(y, axis=axis)),
           tf.split(y, axis=axis)
       ], axis=axis)
     = tf.concatenate([
           tf.log(tf.split(y, axis=axis)),
           tf.split(y, axis=axis)
       ], axis=axis)
     ```

    """

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
        bijectors = bijectors or ()
        self._bijectors = bijectors

        if split_axis is None:
            assert not bijectors, (
                "split_axis has to be defined if parallel has bijectors.")

        self._split_axis = split_axis

        if split_proportions is None:
            self._split_proportions = (1, ) * len(bijectors)

        assert all(
            isinstance(x, int) for x in self._split_proportions
        ), "Every element in split_proportions must be an integer."

        for bijector in bijectors:
            if not bijector._is_injective:  # pylint: disable=protected-access
                raise NotImplementedError(
                    "Invert is not implemented for non-injective bijector ({})"
                    "".format(bijector.name))

        dtype = list(set([b.dtype for b in bijectors]))
        if len(dtype) > 2:
            raise ValueError("incompatible dtypes: %s" % dtype)
        elif len(dtype) == 2:
            dtype = dtype[1] if dtype[0] is None else dtype[0]
        elif len(dtype) == 1:
            dtype = dtype[0]
        else:
            dtype = None

        forward_min_event_ndims = max(
            bijector.forward_min_event_ndims for bijector in bijectors)
        inverse_min_event_ndims = min(
            bijector.forward_min_event_ndims for bijector in bijectors)

        super(Parallel, self).__init__(
            graph_parents=list(itertools.chain.from_iterable(
                b.graph_parents for b in bijectors)),
            forward_min_event_ndims=forward_min_event_ndims,
            inverse_min_event_ndims=inverse_min_event_ndims,
            is_constant_jacobian=all(
                b.is_constant_jacobian for b in bijectors),
            validate_args=validate_args,
            dtype=dtype,
            name=name or (
                "identity" if not bijectors else
                "_and_".join(["parallel"] + [b.name for b in bijectors])))

    @property
    def bijectors(self):
        return self._bijectors

    def _forward(self, x, **kwargs):
        proportions = self._split_proportions
        axis = self._split_axis
        bijectors = self._bijectors

        num_splits = tf.reduce_sum(proportions)
        split_x = tf.split(x, num_splits, axis=axis)

        ys = []
        for i, (bijector, split_size) in enumerate(
                zip(bijectors, proportions)):
            start = sum(proportions[:i])
            y = bijector.forward(
                tf.concat(split_x[start:start+split_size], axis=axis),
                **kwargs.get(bijector.name, {}))
            ys.append(y)

        full_y = tf.concat(ys, axis=axis)

        return full_y

    def _inverse(self, y, **kwargs):
        proportions = self._split_proportions
        axis = self._split_axis
        bijectors = self._bijectors

        num_splits = tf.reduce_sum(proportions)
        split_y = tf.split(y, num_splits, axis=axis)

        xs = []
        for i, (bijector, split_size) in enumerate(
                zip(bijectors, proportions)):
            start = sum(proportions[:i])
            x = bijector.inverse(
                tf.concat(split_y[start:start+split_size], axis=axis),
                **kwargs.get(bijector.name, {}))
            xs.append(x)

        full_x = tf.concat(xs, axis=axis)

        return full_x

    def _forward_log_det_jacobian(self, x, **kwargs):
        x = ops.convert_to_tensor(x, name="x")

        fldj = math_ops.cast(0., dtype=x.dtype.base_dtype)

        if not self.bijectors:
            return fldj

        event_ndims = self._maybe_get_static_event_ndims(
            self.forward_min_event_ndims)

        if _use_static_shape(x, event_ndims):
            event_shape = x.shape[x.shape.ndims - event_ndims:]
        else:
            event_shape = array_ops.shape(x)[array_ops.rank(x) - event_ndims:]

        proportions = self._split_proportions
        axis = self._split_axis
        bijectors = self._bijectors

        num_splits = tf.reduce_sum(proportions)
        split_x = tf.split(x, num_splits, axis=axis)

        ldjs = []
        for i, (bijector, split_size) in enumerate(
                zip(bijectors, proportions)):
            start = sum(proportions[:i])
            ldj = bijector.forward_log_det_jacobian(
                tf.concat(split_x[start:start+split_size], axis=axis),
                event_ndims=event_ndims,
                **kwargs.get(bijector.name, {}))
            ldjs.append(ldj)

        full_ldj = tf.concat(ldjs, axis=axis)

        return full_ldj

    def _inverse_log_det_jacobian(self, y, **kwargs):
        y = ops.convert_to_tensor(y, name="y")
        ildj = math_ops.cast(0., dtype=y.dtype.base_dtype)

        if not self.bijectors:
            return ildj

        event_ndims = self._maybe_get_static_event_ndims(
            self.inverse_min_event_ndims)

        if _use_static_shape(y, event_ndims):
            event_shape = y.shape[y.shape.ndims - event_ndims:]
        else:
            event_shape = array_ops.shape(y)[array_ops.rank(y) - event_ndims:]

        proportions = self._split_proportions
        axis = self._split_axis
        bijectors = self._bijectors

        num_splits = tf.reduce_sum(proportions)
        split_y = tf.split(y, num_splits, axis=axis)

        ildjs = []
        for i, (bijector, split_size) in enumerate(
                zip(bijectors, proportions)):
            start = sum(proportions[:i])
            ildj = bijector.inverse_log_det_jacobian(
                tf.concat(split_y[start:start+split_size], axis=axis),
                event_ndims=event_ndims,
                **kwargs.get(bijector.name, {}))
            ildjs.append(ildj)

        full_ildj = tf.concat(ildjs, axis=axis)

        return full_ildj
