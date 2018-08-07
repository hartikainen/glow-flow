import tensorflow as tf
import tensorflow_probability as tfp
import snapshottest

from glow.bijectors import GlowFlow, GlowStep
from glow.bijectors.squeeze import Squeeze

tfb = tfp.bijectors

tf.enable_eager_execution()
tf.set_random_seed(1)


class TestGlowFlow(tf.test.TestCase, snapshottest.TestCase):

    def setUp(self):
        self.batch_size = 1
        # event_dims = image_size
        self.event_dims = (8, 8, 3)
        tf.reset_default_graph()

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

        z2 = tfb.Chain(list(reversed(
            flow.flow_steps[:2] + flow.flow_steps[-1:]
        ))).forward(x)

        self.assertNotEqual(z, z2)
        self.assertTrue(tf.reduce_all(
            tf.equal(z[..., :event_dims[-1]/2], z2[..., :event_dims[-1]/2])
        ))

    def testVerifyTrainableVariables(self):
        num_levels = 3
        flow = GlowFlow(
            level=num_levels, level_depth=2, validate_args=False)
        event_dims = (64, 64, 8)
        x = tf.random_uniform(
            (self.batch_size, ) + event_dims, dtype=tf.float32)

        flow.forward(x)

        trainable_variables = tf.trainable_variables()

        raise NotImplementedError(
            "Should test that the trainable variables match expectation")

        with self.test_session():
            self.assertMatchSnapshot(trainable_variables)


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
