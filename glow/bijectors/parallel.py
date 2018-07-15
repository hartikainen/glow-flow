"""Parallel bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import tensorflow as tf
import numpy as np


__all__ = [
    "Parallel",
]


tfd = tf.contrib.distributions
tfb = tfd.bijectors


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
        split_proportions = self._split_proportions
        split_axis = self._split_axis
        bijectors = self._bijectors

        num_splits = tf.reduce_sum(split_proportions)
        split_x = tf.split(x, num_splits, axis=split_axis)

        outs = []
        for (i, (bijector, split_size)) in enumerate(
                zip(bijectors[::-1], split_proportions[::-1])):
            start = sum(split_proportions[:i])
            out = bijector.forward(
                split_x[start:start+split_size],
                **kwargs.get(bijector.name, {}))
            outs.append(out)

        full_out = tf.concat([outs], axis=split_axis)

        return full_out

    def _inverse(self, y, **kwargs):
        split_proportions = self._split_proportions
        split_axis = self._split_axis
        bijectors = self._bijectors

        num_splits = tf.reduce_sum(split_proportions)
        split_x = tf.split(x, num_splits, axis=split_axis)

        outs = []
        for (i, (bijector, split_size)) in enumerate(
                zip(bijectors, split_proportions)):
            start = sum(split_proportions[:i])
            out = bijector.forward(
                split_x[start:start+split_size],
                **kwargs.get(bijector.name, {}))
            outs.append(out)

        full_out = tf.concat([outs], axis=split_axis)

        return full_out

    def _forward_log_det_jacobian(self, x):
        raise NotImplementedError('_forward_log_det_jacobian')

    def _inverse_log_det_jacobian(self, y):
        raise NotImplementedError('_inverse_log_det_jacobian')
