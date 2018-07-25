import tensorflow as tf
import snapshottest

from glow.bijectors import Squeeze


tf.enable_eager_execution()
tf.set_random_seed(1)


class TestSqueeze(tf.test.TestCase, snapshottest.TestCase):

    def setUp(self):
        self.batch_size = 1
        self.forward_event_dims = (4,4,2)
        self.inverse_event_dims = (2,2,8)
        self.event_ndims = len(self.forward_event_dims)
        self.squeeze = Squeeze(factor=2)

    def testForward(self):
        dims = (self.batch_size, ) + self.forward_event_dims
        x = tf.reshape(tf.range(tf.reduce_prod(dims)), dims)

        z = self.squeeze.forward(x)

        with self.test_session():
            (H, W, C) = x.shape[-3:]
            factor = self.squeeze.factor
            self.assertEqual(z.shape[-3:], (H//factor, W//factor, C*factor**2))
            self.assertMatchSnapshot(z.numpy().tolist())

    def testInverse(self):
        dims = (self.batch_size, ) + self.inverse_event_dims
        z = tf.reshape(tf.range(tf.reduce_prod(dims)), dims)

        x = self.squeeze.inverse(z)

        with self.test_session():
            (H, W, C) = z.shape[-3:]
            factor = self.squeeze.factor
            self.assertEqual(x.shape[-3:], (H*factor, W*factor, C//factor**2))
            self.assertMatchSnapshot(x.numpy().tolist())

    def testForwardLogDetJacobian(self):
        dims = (self.batch_size, ) + self.forward_event_dims
        x = tf.reshape(tf.range(tf.reduce_prod(dims)), dims)

        log_det_jacobian = self.squeeze.forward_log_det_jacobian(
            x, event_ndims=self.event_ndims)
        expected = tf.constant(0.0)

        with self.test_session():
            self.assertAllEqual(log_det_jacobian.numpy(), expected.numpy())

    def testInverseLogDetJacobian(self):
        dims = (self.batch_size, ) + self.inverse_event_dims
        z = tf.reshape(tf.range(tf.reduce_prod(dims)), dims)

        inverse_log_det_jacobian = self.squeeze.inverse_log_det_jacobian(
            z, event_ndims=self.event_ndims)
        expected = tf.constant(0.0)

        with self.test_session():
            self.assertAllEqual(
                inverse_log_det_jacobian.numpy(), expected.numpy())

    def testBijective(self):
        squeeze = Squeeze(
            validate_args=False,
            forward_min_event_ndims=1,
            inverse_min_event_ndims=1,
        )
        x = tf.random_uniform(
            (self.batch_size, ) + self.forward_event_dims, dtype=tf.float32)
        x_ = squeeze.inverse(tf.identity(squeeze.forward(x)))

        with self.test_session():
            self.assertAllEqual(x, x_.numpy())


if __name__ == '__main__':
    tf.test.main()
