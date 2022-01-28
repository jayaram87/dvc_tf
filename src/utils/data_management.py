import tensorflow as tf
import logging
import os
import numpy as np


def train_valid_generator(data_dir: str = 'data') -> tuple:
    x_valid = np.load(os.path.join(data_dir, 'X_train_full.npy'))[:5000] / 255
    x_train = np.load(os.path.join(data_dir, 'X_train_full.npy'))[5000:] / 255
    y_valid = np.load(os.path.join(data_dir, 'y_train_full.npy'))[:5000]
    y_train = np.load(os.path.join(data_dir, 'y_train_full.npy'))[5000:]

    logging.info("train and valid data created is created.")
    return x_valid, x_train, y_valid, y_train