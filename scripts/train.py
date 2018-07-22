from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import signal
from datetime import datetime

# Dependency imports
import tensorflow as tf
import tensorflow_probability as tfp

from .datasets import get_input_fns
from glow.bijectors import GlowFlow

tfd = tfp.distributions


AVAILABLE_DATASETS = ('mnist', )

TMP_DIR = os.getenv("TEST_TMP_DIR", "/tmp")


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

    parser.add_argument(
        '--data_dir',
        type=str,
        default=os.path.join(TMP_DIR, "data/"),
        help="Base directory for the data.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.path.join(
            TMP_DIR,
            "data/models/",
            datetime.now().strftime('%Y%m%dT%H%M%S')),
        help="Directory to save the model.")

    parser.add_argument("--visualize_every",
                        type=int,
                        default=500,
                        help="Frequency at which to save visualizations.")

    parser.add_argument("--n_train",
                        type=int,
                        default=50000,
                        help="Training epoch size.")

    parser.add_argument("--n_test",
                        type=int,
                        default=-1,
                        help="Validation epoch size.")

    parser.add_argument("--batch_size",
                        type=int,
                        default=64,
                        help="Batch size.")

    parser.add_argument("--max_steps",
                        type=int,
                        default=int(1e6),
                        help="Total number of training steps to run.")

    parser.add_argument(
        "--activation",
        type=lambda activation: getattr(tf.nn, activation),
        default="leaky_relu",
        help="Activation function for all hidden layers.")

    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_levels", type=int, default=3,
                        help="Number of Glow levels in the flow.")
    parser.add_argument("--level_depth", type=int, default=3,
                        help="Number of flow steps in each Glow level.")

    args = parser.parse_args()

    return args


def model_fn(features, labels, mode, params, config):
    """Build the glow flow model function for use in an estimator.

    Arguments:
        features: The input features for the estimator.
        labels: The labels, unused here.
        mode: Signifies whether it is train or test or predict.
        params: Some hyperparameters as a dictionary.
        config: The RunConfig, unused here.
        Returns:
        EstimatorSpec: A tf.estimator.EstimatorSpec instance.
    """
    # TODO: Implement this


def train_model(args):
    # TODO: make these tensorflow datasets or something more efficient
    train_input_fn, eval_input_fn = get_input_fns(
        args.dataset, args.data_dir, args.batch_size)

    estimator = tf.estimator.Estimator(
        model_fn,
        params=vars(args),
        config=tf.estimator.RunConfig(
            model_dir=args.model_dir,
            save_checkpoints_steps=args.visualize_every))

    for _ in range(args.max_steps // args.visualize_every):
        estimator.train(train_input_fn, steps=args.visualize_every)
        eval_results = estimator.evaluate(eval_input_fn)
        print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__ == '__main__':
    # This enables a ctr-C without triggering errors
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    args = parse_args()

    train_model(args)
