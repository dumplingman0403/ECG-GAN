import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, UpSampling1D
from tensorflow.keras.layers import Conv1DTranspose, Conv1D, Bidirectional, LSTM
from tensorflow.keras.layers import LeakyReLU, MaxPooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam



class Generator:
    def __init__(self, latent_size):
        self.latent_size = latent_size

    def G_vl(self):

        model = Sequential(name='Generator_v1')
        model.add(Reshape((self.latent_size, 1)))
        model.add(Bidirectional(LSTM(16, return_sequences=True)))
        # model.add(Bidirectional(LSTM(64)))
        # model.add(Flatten())
        # model.add(UpSampling1D())
        model.add(Conv1D(32, kernel_size=8, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling1D())
        model.add(Conv1D(16, kernel_size=8, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(UpSampling1D())
        model.add(Conv1D(8, kernel_size=8, padding="same"))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv1D(1, kernel_size=8, padding="same"))
        model.add(Flatten())

        # model.add(Dense(100))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(Dense(150))
        # model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.input_shape[0]))
        model.add(Activation('tanh'))
        model.add(Reshape(self.input_shape))
        noise = Input(shape=(self.latent_size,))
        signal = model(noise)

        model.summary()

        return Model(inputs=noise, outputs=signal) 

    def G_v2(self):
        model = Sequential(name='Generator_v2')
        model.add(Reshape((self.latent_size, 1)))
        model.add(Bidirectional(LSTM(1, return_sequences=True)))
        model.add(Flatten())
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(150))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(self.input_shape[0]))
        model.add(Activation('tanh'))
        model.add(Reshape(self.input_shape))
        noise = Input(shape=(self.latent_size,))
        signal = model(noise)

        model.summary()

        return Model(inputs=noise, outputs=signal) 

    def G_v3(self):
        model = Sequential(name='Generator')
        model.add(Reshape((self.latent_size, 1)))
        model.add(Bidirectional(LSTM(16, return_sequences=True)))
        model.add(Bidirectional(LSTM(16, return_sequences=True)))
        model.add(Flatten())
        model.add(Dense(self.input_shape[0]))
        model.add(Activation('tanh'))
        model.add(Reshape(self.input_shape))
        noise = Input(shape=(self.latent_size,))
        signal = model(noise)

        model.summary()
        return Model(inputs=noise, outputs=signal) 