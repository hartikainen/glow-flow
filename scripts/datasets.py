import numpy as np
import tensorflow as tf
from tensorflow.python import keras


def get_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    y_train = np.reshape(y_train, [-1])
    y_test = np.reshape(y_test, [-1])
    # Pad with zeros to make 32x32
    x_train = np.lib.pad(x_train, ((0, 0), (2, 2), (2, 2)), 'minimum')
    # Pad with zeros to make 32x23
    x_test = np.lib.pad(x_test, ((0, 0), (2, 2), (2, 2)), 'minimum')
    x_train = np.tile(np.reshape(x_train, (-1, 32, 32, 1)), (1, 1, 1, 3))
    x_test = np.tile(np.reshape(x_test, (-1, 32, 32, 1)), (1, 1, 1, 3))

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    return (x_train, y_train), (x_test, y_test)

DATASET_LOADERS = {
    'mnist': get_mnist
}

def get_dataset(dataset_name):
    return DATASET_LOADERS[dataset_name]()
