import numpy as np
import tensorflow as tf
import snapshottest

from glow.bijectors import GlowFlow


tf.enable_eager_execution()
tf.set_random_seed(1)


class TestGlowFlow(tf.test.TestCase, snapshottest.TestCase):

    def setUp(self):
        self.batch_size = batch_size = 1
        # event_dims = image_size
        self.event_dims = event_dims = (8,8,4)

    def testBijective(self):
        flow = GlowFlow(
            num_levels=2,
            level_depth=2,
            validate_args=False)
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        x_ = flow.inverse(flow.forward(x))

        with self.test_session():
            self.assertAllEqual(x, x_.numpy())


if __name__ == '__main__':
  tf.test.main()
