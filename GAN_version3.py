import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, UpSampling1D
from tensorflow.keras.layers import Conv1DTranspose, Conv1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import pickle
class DCGAN:

    def __init__(self, input_shape=(180, 1), latent_size=100):
        
        self.input_shape = input_shape
        self.latent_size = latent_size
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.discrimintor = self.bulid_discrimintor()
        self.discrimintor.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.generator = self.build_generator()
        z = Input(shape=(self.latent_size, ))
        signal = self.generator(z)

        self.discrimintor.trainable = False

        valid = self.discrimintor(signal)

        self.combine = Model(z, valid)
        self.combine.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        
        model = Sequential(name='Generator')
        model.add(Reshape((self.latent_size, 1)))
        model.add(Conv1D(256, kernel_size=4, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(UpSampling1D())
        model.add(Conv1D(128, kernel_size=4, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(UpSampling1D())
        model.add(Conv1D(64, kernel_size=4, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(UpSampling1D())

        model.add(Conv1D(32, kernel_size=4, strides=2,padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Conv1D(16, kernel_size=4, strides=2, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Conv1D(1, kernel_size=4, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(180 * 1))
        model.add(Activation('tanh'))
        model.add(Reshape(self.input_shape))
        noise = Input(shape=(self.latent_size,))
        signal = model(noise)

        model.summary()

        return Model(inputs=noise, outputs=signal) 


        

    def bulid_discrimintor(self):
        model = Sequential(name='Discriminator')
        model.add(Conv1D(32, kernel_size=4, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(64, kernel_size=4, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv1D(128, kernel_size=4, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Conv1D(256, kernel_size=4, strides=2, input_shape=self.input_shape, padding='same'))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        
        signal = Input(shape=self.input_shape)
        validity = model(signal)

        return Model(inputs=signal, outputs=validity)
        

    def gen_latent(self, batch_size):
        return np.random.normal(0, 1, size=(batch_size, self.latent_size))
    
    def train(self, epochs, X_train, batch_size=128, save_interval=50):
        vaild = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))        
        
        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            signals = X_train[idx]
            noise = self.gen_latent(batch_size)

            gen_signals = self.generator.predict(noise)

            d_loss_real = self.discrimintor.train_on_batch(signals, vaild)
            d_loss_fake = self.discrimintor.train_on_batch(gen_signals, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            g_loss = self.combine.train_on_batch(noise, vaild)

            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % save_interval == 0:
                self.save_image(epoch)

    def save_image(self, epoch):

        if os.path.isdir('image/') != True:
            os.mkdir('image/')

        r, c = 3, 3
        noise = self.gen_latent(r*c)

        signals = self.generator(noise)

        fig, axs = plt.subplots(r, c)
        cnt=0
        for i in range(r):
            for j in range(c):
                axs[i, j].plot(signals[cnt])
                cnt += 1

        fig.savefig('image/ecg_sig_%d.png' %epoch )
        plt.close()

    def prepare_input(self, dataset):

        if type(dataset) != dict:
            raise TypeError('Dateset type must be dictionary.')
        X_train = []
        y = []
        for sg_id in dataset.keys():
            lb = str(dataset[sg_id][0])
            if lb != '~':
                signal = dataset[sg_id][1]
                for hb in signal:
                    X_train.append(hb)
                    y.append(lb)
        
        X_train = np.array(X_train).reshape(-1, X_train.shape[1], 1)
        y = np.array(y)
        return X_train, y





