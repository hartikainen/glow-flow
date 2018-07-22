import tensorflow as tf
import snapshottest

from glow.bijectors import ConvolutionPermute


tf.enable_eager_execution()
tf.set_random_seed(1)


class TestConvolutionPermute(tf.test.TestCase, snapshottest.TestCase):

    def setUp(self):
        self.batch_size = 1
        self.event_dims = (4, 4, 3)
        self.event_ndims = len(self.event_dims)
        self.bijector = ConvolutionPermute(validate_args=False)

    def testForward(self):
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)

        z = self.bijector.forward(x)

        with self.test_session():
            self.assertMatchSnapshot(z.numpy().tolist())

    def testInverse(self):
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)

        z = self.bijector.inverse(x)

        with self.test_session():
            self.assertMatchSnapshot(z.numpy().tolist())

    def testForwardLogDetJacobian(self):
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)

        log_det_jacobian = self.bijector.forward_log_det_jacobian(
            x, event_ndims=self.event_ndims)

        with self.test_session():
            self.assertMatchSnapshot(log_det_jacobian.numpy().tolist())

    def testInverseLogDetJacobian(self):
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)

        inverse_log_det_jacobian = self.bijector.inverse_log_det_jacobian(
            x, event_ndims=self.event_ndims)

        with self.test_session():
            self.assertMatchSnapshot(inverse_log_det_jacobian.numpy().tolist())

    def testBijective(self):
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        x_ = self.bijector.inverse(self.bijector.forward(x))

        with self.test_session():
            self.assertAllEqual(x, x_.numpy())


if __name__ == '__main__':
    tf.test.main()
