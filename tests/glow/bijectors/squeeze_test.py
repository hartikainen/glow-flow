import numpy as np
import tensorflow as tf
import snapshottest

from glow.bijectors import Squeeze


tf.enable_eager_execution()
tf.set_random_seed(1)


class TestSqueeze(tf.test.TestCase, snapshottest.TestCase):

    def setUp(self):
        self.batch_size = batch_size = 1
        self.event_dims = event_dims = (2,2,3)

    def testBijective(self):
        squeeze = Squeeze(validate_args=False)
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        x_ = squeeze.inverse(squeeze.forward(x))

        with self.test_session():
            self.assertAllEqual(x, x_.numpy())


if __name__ == '__main__':
  tf.test.main()
