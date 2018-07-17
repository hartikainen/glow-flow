import argparse
import signal

import tensorflow as tf

from .datasets import get_dataset
from glow.bijectors import GlowFlow


tfd = tf.contrib.distributions

AVAILABLE_DATASETS = ('mnist')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',
                        type=str,
                        default='.',
                        help="Base directory for the logs.")

    parser.add_argument('--dataset',
                        type=str,
                        choices=AVAILABLE_DATASETS,
                        default='mnist')

    parser.add_argument('--dataset_dir',
                        type=str,
                        default='.',
                        help="Base directory for the data.")

    parser.add_argument("--n_train",
                        type=int,
                        default=50000,
                        help="Training epoch size.")

    parser.add_argument("--n_test",
                        type=int,
                        default=-1,
                        help="Validation epoch size.")

    parser.add_argument("--train_batch_size",
                        type=int,
                        default=64,
                        help="Training batch size.")

    parser.add_argument("--test_batch_size",
                        type=int,
                        default=50,
                        help="Test batch size.")

    parser.add_argument("--num_epochs",
                        type=int,
                        default=int(1e6),
                        help="Total number of training epochs.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_levels", type=int, default=3,
                        help="Number of Glow levels in the flow.")
    parser.add_argument("--level_depth", type=int, default=3,
                        help="Number of flow steps in each Glow level.")

    args = parser.parse_args()

    return args


def train_model(args):
    # TODO: make these tensorflow datasets or something more efficient
    (x_train, y_train), (x_test, y_test) = get_dataset(args.dataset)

    glow_flow = GlowFlow(
        num_levels=args.num_levels,
        level_depth=args.level_depth)

    base_distribution = tfd.MultivariateNormalDiag(
        loc=tf.zeros(x_train.shape[1:]),
        scale_diag=tf.ones(x_train.shape[1:]))

    transformed_glow_flow = tfd.ConditionalTransformedDistribution(
        distribution=base_distribution,
        bijector=glow_flow,
        name="transformed_glow_flow")

    z = glow_flow.forward(x_train[:5])
    x_ = glow_flow.inverse(z)
    samples = transformed_glow_flow.sample(5)


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    args = parse_args()

    train_model(args)
