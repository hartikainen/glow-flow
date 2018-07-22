import numpy as np
import tensorflow as tf
from tensorflow.python import keras


def load_mnist(data_dir, batch_size):
    """Build an Iterator switching between train and holdout data."""

    (x_train, y_train), (x_eval, y_eval) = keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)[..., np.newaxis]
    y_train = y_train.astype(np.float32)
    x_eval = x_eval.astype(np.float32)[..., np.newaxis]
    y_eval = y_eval.astype(np.float32)

    x_train = np.repeat(x_train, 3, axis=-1)
    x_eval = np.repeat(x_eval, 3, axis=-1)

    assert x_train.shape[-1] == 3

    # TODO: reshaping, padding?

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = (
        train_dataset.shuffle(50000).repeat().batch(batch_size))

    def train_input_fn():
        return train_dataset.make_one_shot_iterator().get_next()

    eval_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    eval_dataset = eval_dataset.batch(batch_size)

    def eval_input_fn():
        return eval_dataset.make_one_shot_iterator().get_next()

    return train_input_fn, eval_input_fn


DATASET_LOADERS = {
    'mnist': load_mnist
}


def get_input_fns(dataset_name, data_dir, batch_size):
    return DATASET_LOADERS[dataset_name](data_dir, batch_size)
