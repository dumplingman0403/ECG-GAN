import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import backend as K
import utils as ut
from tqdm import tqdm
from matplotlib import pyplot

LATENT_DIM = (23, 12)


def define_generator(in_shape):

    g_model = tf.keras.Sequential()
    g_model.add(layers.Input(shape=in_shape))

    g_model.add(layers.Conv1D(128, 16, padding='same'))
    g_model.add(layers.LeakyReLU())
    g_model.add(layers.UpSampling1D(2))
    g_model.add(layers.BatchNormalization())

    g_model.add(layers.Conv1D(64, 16, padding='same'))
    g_model.add(layers.LeakyReLU())
    g_model.add(layers.UpSampling1D(2))
    g_model.add(layers.BatchNormalization())

    g_model.add(layers.Conv1D(32, 16, padding='same'))
    g_model.add(layers.LeakyReLU())
    g_model.add(layers.UpSampling1D(2))
    g_model.add(layers.BatchNormalization())

    g_model.add(layers.Conv1D(1, 16, activation='tanh',padding='same'))
    g_model.summary()
    return g_model

def define_discriminator(in_shape):

    d_model = tf.keras.Sequential()
    d_model.add(layers.Input(shape=in_shape))
    d_model.add(layers.Conv1D(32, 16,  padding='same'))
    d_model.add(layers.LeakyReLU())
    d_model.add(layers.BatchNormalization())
    d_model.add(layers.MaxPooling1D())

    d_model.add(layers.Conv1D(64, 16,  padding='same'))
    d_model.add(layers.LeakyReLU())
    d_model.add(layers.BatchNormalization())
    d_model.add(layers.MaxPooling1D())

    d_model.add(layers.Conv1D(128, 16,  padding='same'))
    d_model.add(layers.LeakyReLU())
    d_model.add(layers.BatchNormalization())
    d_model.add(layers.MaxPooling1D())

    d_model.add(layers.Flatten())
    d_model.add(layers.Dense(1000))
    d_model.add(layers.Dense(4))
    d_model.summary()
    return d_model


def generate_latent(dim, n_sample, n_class):
    z = np.random.normal(0, 1, size=(n_sample, dim))
    z_label = np.random.randint(0, n_class, size=(n_sample, 1))
    y_label = np.repeat(n_class + 1, n_sample).reshape(n_sample, 1)
    return [z, z_label], y_label




"""
test
"""
if __name__ == "__main__":
    # gen = define_generator((23, 12))
    dis = define_discriminator((180, 1))