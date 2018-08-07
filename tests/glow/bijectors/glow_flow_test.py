import tensorflow as tf
import snapshottest

from glow.bijectors import GlowFlow, GlowStep
from glow.bijectors.squeeze import Squeeze


tf.enable_eager_execution()
tf.set_random_seed(1)


class TestGlowFlow(tf.test.TestCase, snapshottest.TestCase):

    def setUp(self):
        self.batch_size = 1
        # event_dims = image_size
        self.event_dims = (8, 8, 3)

    def testBijective(self):
        flow = GlowFlow(level=3, level_depth=3, validate_args=False)
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        z = flow.inverse(x)
        x_ = flow.forward(tf.identity(z))

        assert z.shape == x.shape

        with self.test_session():
            self.assertAllEqual(x, x_.numpy())

    def testForward(self):
        flow = GlowFlow(level=3, level_depth=3, validate_args=False)
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        z = flow.forward(x)

        # with self.test_session():
        #     self.assertMatchSnapshot(x.numpy().tolist())

    def testInverse(self):
        flow = GlowFlow(level=3, level_depth=3, validate_args=False)
        x = tf.random_uniform(
            (self.batch_size, ) + self.event_dims, dtype=tf.float32)
        z = flow.inverse(x)

        # with self.test_session():
        #     self.assertMatchSnapshot(z.numpy().tolist())

    def testSingleLevelShapes(self):
        num_levels = 1
        flow = GlowFlow(
            level=num_levels, level_depth=2, validate_args=False)
        event_dims = (64, 64, 8)
        x = tf.random_uniform(
            (self.batch_size, ) + event_dims, dtype=tf.float32)
        z = flow.forward(x)

        self.assertEqual(x.shape, z.shape)

    def testParallelPassthrough(self):
        num_levels = 3
        flow = GlowFlow(
            level=num_levels, level_depth=2, validate_args=False)
        event_dims = (64, 64, 8)
        x = tf.random_uniform(
            (self.batch_size, ) + event_dims, dtype=tf.float32)
        z = flow.forward(x)

        for level in range(num_levels):
            # TODO: Test somehow that all the passthrough values match
            # expectation
            pass

    def testVerifyTrainableVariables(self):
        raise NotImplementedError(
            "Should test that the trainable variables match expectation")


class TestGlowStep(tf.test.TestCase, snapshottest.TestCase):

    def setUp(self):
        self.batch_size = 1
        # event_dims = image_size
        self.event_dims = (8, 8, 3)

    def testForward(self):
        step = GlowStep(depth=2, validate_args=False)
        x = Squeeze(factor=2).forward(
            tf.random_uniform(
                (self.batch_size, ) + self.event_dims, dtype=tf.float32))
        z = step.forward(x)

        self.assertEqual(x.shape, z.shape)

        # with self.test_session():
        #     self.assertMatchSnapshot(x.numpy().tolist())

    def testVerifyTrainableVariables(self):
        raise NotImplementedError(
            "Should test that the trainable variables match expectation")


if __name__ == '__main__':
    tf.test.main()
