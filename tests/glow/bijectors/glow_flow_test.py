import tensorflow as tf
import snapshottest

from glow.bijectors import GlowFlow
from glow.bijectors.squeeze import Squeeze


tf.enable_eager_execution()
tf.set_random_seed(1)


class TestGlowFlow(tf.test.TestCase, snapshottest.TestCase):

    def setUp(self):
        self.batch_size = 1
        # event_dims = image_size
        self.event_dims = (32, 32, 3)

    def testBijective(self):
        flow = GlowFlow(
            num_levels=2,
            level_depth=2,
            validate_args=False)
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        # Squeeze to get even last dimension size
        x = Squeeze(factor=2).forward(x)
        z = flow.inverse(x)
        x_ = flow.forward(tf.identity(z))

        assert z.shape == x.shape


        with self.test_session():
            self.assertAllEqual(x, x_.numpy())


if __name__ == '__main__':
    tf.test.main()
