import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, UpSampling1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

class DCGAN:

    def __init__(self, input_shape, latent_size=100):
        
        self.input_shape = input_shape
        self.latent_size = latent_size


    def generator(self):
        
        model = Sequential()
        model.add(Dense(128, activation='relu', input_dim=self.latent_size))
        
        noise = Input(shape=(self.latent_size,))
        signal = model(noise)

        return Model(inputs=noise, outputs=signal) 


        

    def discrimintor(self):

        pass

    def gen_latent(self):

        return np.random.normal(0, 1, size=self.latent_size)
    
