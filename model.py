"""
coding: utf-8

author: Eric Wu
"""

"""
to do:

1. input ecg normalize range [-1, 1]
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from numpy.random import randn
import tensorflow_gan
from tensorflow.keras import backend as K
NUM_LATENT = 100
NUM_BATCH = 32
NUM_DATA = 8528
NUM_CLASS = 4              #data info: Normal:5154, AF:771, OtherRhythm:2557, Noisy:46, Totol:8525 
DATA_SHAPE = (NUM_DATA, 180, 1)
class TSGAN(object):
    
    def __init__(self):
        self.latent_size = NUM_LATENT
        self.batch_size = NUM_BATCH
        self.realdata_size = NUM_DATA
        self.num_class = NUM_CLASS
        self.realdata_shape = DATA_SHAPE

    def set_generator(self):    #this part need to be redesign
                                #Use ReLU in all layers except output use tanh
        #label input
        
        # in_l = tf.keras.Input(shape=(1,))

        #latent input
        in_lat = tf.keras.Input(shape=(100, ))
        nodes = 45*12
        lat_lyr = layers.Dense(nodes)(in_lat)
        lat_lyr = layers.Reshape((45, 12))(lat_lyr)
        lat_lyr = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(lat_lyr)
        lat_lyr = layers.Conv1D(32, 3, activation='relu', padding='same')(lat_lyr)
        lat_lyr = layers.UpSampling1D(size=2)(lat_lyr)
        lat_lyr = layers.Conv1D(16, 3, activation='relu', padding='same')(lat_lyr)
        lat_lyr = layers.UpSampling1D(size=2)(lat_lyr)
        lat_lyr = layers.Conv1D(1, 3, activation='tanh', padding='same')(lat_lyr)

        model = tf.keras.Model(in_lat, lat_lyr)
        
        return model
        
    def set_discriminator(self):     #use leaky ReLU activation in all layers
        input_ecg = tf.keras.Input(shape=(self.realdata_shape[1:]))

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
        out1 = layers.Dense(self.num_class, activation='softmax')(lyr)
        # real/fake output
        out2 = layers.Dense(1, activation='sigmoid')(lyr)

        model = tf.keras.Model(input_ecg, [out1, out2])
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['categorical_crossentropy', 'binary_crossentropy'], optimizer=opt)
        # model.summary()
        
        return model

    def generate_latent(self):
        #generate latent input
        l_input = randn(self.latent_size * self.realdata_size)
        #reshape latent input to num_data*latent_size 
        l_input = np.reshape(l_input, (self.realdata_size, self.latent_size))
        label = np.random.randint(0, self.num_class, self.realdata_size)
        return [l_input, label]

    def generate_fake_input(self, generator):
        l_input, label_input = self.generate_latent()
        ecg = generator.predict([l_input, label_input])
        y = np.zeros((self.realdata_size, 1))               #fake label 0
        return [ecg, label_input], y 

    def set_GAN(self, g_model, d_model):

        d_model.trainable = False

        gan_output= d_model(g_model.output)

        model = tf.keras.Model(g_model.input, gan_output)
        opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['categorical_crossentropy','binary_crossentropy'], optimizer=opt)
        return model

    def get_parameter(self):
        pass 

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)