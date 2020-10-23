import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randn
from tensorflow.keras import backend as K



def discriminator(input_shape, num_class):
    input_ecg = tf.keras.Input(shape=input_shape)

    lyr = layers.Conv1D(8, 3, activation=layers.LeakyReLU(alpha=0.2), padding='same')(input_ecg)
    lyr = layers.BatchNormalization()(lyr)
    lyr = layers.MaxPooling1D()(lyr)

    lyr = layers.Conv1D(16, 3, activation=layers.LeakyReLU(alpha=0.2), padding='same')(lyr)
    lyr = layers.BatchNormalization()(lyr)
    lyr = layers.MaxPooling1D()(lyr)

    lyr = layers.Conv1D(32, 3, activation=layers.LeakyReLU(alpha=0.2), padding='same')(lyr)
    lyr = layers.BatchNormalization()(lyr)
    lyr = layers.MaxPooling1D()(lyr)

    lyr = layers.GRU(30)(lyr)
    lyr = layers.Dropout(0.5)(lyr)
    lyr = layers.GaussianNoise(0.2)(lyr)
    

    # class label output  
    out1 = layers.Dense(num_class, activation='softmax')(lyr)
    # real/fake output
    out2 = layers.Dense(1, activation='sigmoid')(lyr)

    model = tf.keras.Model(input_ecg, [out1, out2])
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], optimizer=opt)
    # model.summary()
    
    return model

