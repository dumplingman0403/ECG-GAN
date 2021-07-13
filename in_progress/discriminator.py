import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, UpSampling1D
from tensorflow.keras.layers import Conv1DTranspose, Conv1D, Bidirectional, LSTM
from tensorflow.keras.layers import LeakyReLU, MaxPooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import pickle



class Discriminator():

    def __init__(self, input_shape):
        self.input_shape = input_shape

    def D_v1(self):

        model = Sequential(name='Discriminator_v1')
        model.add(Conv1D(8, kernel_size=3, strides=1, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(MaxPooling1D(3))
        
        model.add(Conv1D(16, kernel_size=3, strides=1, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(MaxPooling1D(3, strides=2))

        model.add(Conv1D(32, kernel_size=3, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(MaxPooling1D(3, strides=2))

        model.add(Conv1D(64, kernel_size=3, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(MaxPooling1D(3, strides=2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        
        signal = Input(shape=self.input_shape)
        validity = model(signal)

        return Model(inputs=signal, outputs=validity)

    def D_v2(self):

        model = Sequential(name='Discriminator_v2')
        model.add(Conv1D(8, kernel_size=3, strides=1, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(MaxPooling1D(3))
        
        model.add(Conv1D(16, kernel_size=3, strides=1, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(MaxPooling1D(3, strides=2))

        model.add(Conv1D(32, kernel_size=3, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(MaxPooling1D(3, strides=2))

        model.add(Conv1D(64, kernel_size=3, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # model.add(Dropout(0.25))
        model.add(MaxPooling1D(3, strides=2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        
        signal = Input(shape=self.input_shape)
        validity = model(signal)

        return Model(inputs=signal, outputs=validity)

