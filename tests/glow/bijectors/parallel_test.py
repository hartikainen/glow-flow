import tensorflow as tf
import snapshottest

from glow.bijectors import Parallel

tfd = tf.contrib.distributions
tfb = tfd.bijectors


tf.enable_eager_execution()
tf.set_random_seed(1)


class TestParallel(tf.test.TestCase, snapshottest.TestCase):

    def setUp(self):
        self.batch_size = 1
        self.event_dims = (2, 2, 3)

    def testForward(self):
        axis = 1
        bijectors = [tfb.Exp(), tfb.Softplus()]
        parallel = Parallel(
            bijectors=bijectors, split_axis=axis, validate_args=False)
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        z_ = parallel.forward(x)

        with self.test_session():
            for i, bijector in enumerate(bijectors):
                self.assertAllEqual(
                    tf.gather(z_, i, axis=axis),
                    bijector.forward(tf.gather(x, i, axis=axis)))

    def testInverse(self):
        axis = 1
        bijectors = [tfb.Exp(), tfb.Softplus()]
        parallel = Parallel(
            bijectors=bijectors, split_axis=axis, validate_args=False)
        z = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        x_ = parallel.inverse(z)

        with self.test_session():
            for i, bijector in enumerate(bijectors):
                self.assertAllEqual(
                    tf.gather(x_, i, axis=axis),
                    bijector.inverse(tf.gather(z, i, axis=axis)))

    def testForwardLogDetJacobian(self):
        axis = 1
        bijectors = [tfb.Exp(), tfb.Softplus()]
        parallel = Parallel(
            bijectors=bijectors, split_axis=1, validate_args=False)
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float64)

        fldj = parallel.forward_log_det_jacobian(
            x, event_ndims=len(self.event_dims))

        expected_fldj = tf.reduce_sum([
            bijector.forward_log_det_jacobian(
                tf.gather(x, i, axis=axis),
                event_ndims=3)
            for i, bijector in enumerate(bijectors)
        ], keep_dims=True)

        with self.test_session():
            self.assertAllClose(fldj, expected_fldj)

    def testInverseLogDetJacobian(self):
        axis = 1
        bijectors = [tfb.Exp(), tfb.Softplus()]
        parallel = Parallel(
            bijectors=bijectors, split_axis=1, validate_args=False)
        z = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float64)

        ildj = parallel.inverse_log_det_jacobian(
            z, event_ndims=len(self.event_dims))

        expected_ildj = tf.reduce_sum([
            bijector.inverse_log_det_jacobian(
                tf.gather(z, i, axis=axis),
                event_ndims=3)
            for i, bijector in enumerate(bijectors)
        ], keep_dims=True)

        with self.test_session():
            self.assertAllClose(ildj, expected_ildj)

    def testBijective(self):
        bijectors = [tfb.Exp(), tfb.Softplus()]
        parallel = Parallel(
            bijectors=bijectors, split_axis=1, validate_args=False)
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        x_ = parallel.inverse(tf.identity(parallel.forward(x)))

        with self.test_session():
            self.assertAllEqual(x, x_.numpy())

    def testUnevenSplitProportions(self):
        axis = 1
        bijectors = [tfb.Exp(), tfb.Softplus()]
        proportions = [1, 2]
        parallel = Parallel(
            bijectors=bijectors,
            split_axis=axis,
            split_proportions=proportions,
            validate_args=False)
        event_dims = (3, 2, 3)
        x = tf.random_uniform(
            (self.batch_size, ) + event_dims, dtype=tf.float32)
        z_ = parallel.forward(x)

        with self.test_session():
            for i, bijector in enumerate(bijectors):
                i_start = sum(proportions[:i])
                i_end = i_start + proportions[i]
                self.assertAllEqual(
                    tf.gather(z_, tf.range(i_start, i_end), axis=axis),
                    bijector.forward(
                        tf.gather(x, tf.range(i_start, i_end), axis=axis)))


if __name__ == '__main__':
    tf.test.main()
