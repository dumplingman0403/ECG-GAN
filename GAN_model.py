import imp
from operator import mod
from tokenize import Name
import numpy as np
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization, Activation, UpSampling1D
from tensorflow.keras.layers import Conv1DTranspose, Conv1D, Bidirectional, LSTM
from tensorflow.keras.layers import LeakyReLU, MaxPooling1D, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import pickle
from Minibatchdiscrimination import MinibatchDiscrimination
import h5py

class DCGAN:

    def __init__(self, input_shape=(180, 1), latent_size=100, random_sine = True, scale=1, minibatch=False):
        
        self.input_shape = input_shape
        self.latent_size = latent_size
        optimizer = Adam(lr=0.0002, beta_1=0.5)
        self.random_sine = random_sine
        self.scale = scale
        self.minibatch = minibatch
        # build and compile discriminator
        self.discrimintor = self.bulid_discrimintor()
        self.discrimintor.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # build generator
        self.generator = self.build_generator()
        
        # generator takes noise as input and generates signals
        z = Input(shape=(self.latent_size, ))
        signal = self.generator(z)

        # for combined model, we only train the generator 
        self.discrimintor.trainable = False
        
        # discrimator takes generate signals as input and determines validity
        valid = self.discrimintor(signal)

        # combine model, stack generator and discrimnator
        # train the generator to fool discriminator 
        self.combine = Model(z, valid)
        self.combine.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        
        model = Sequential()
        model = Sequential(name='Generator')
        model.add(Reshape((self.latent_size, 1)))
        model.add(Bidirectional(LSTM(16, return_sequences=True)))
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


        

    def bulid_discrimintor(self):

        signal = Input(shape=self.input_shape)
        
        if self.minibatch:
            
            flat = Flatten()(signal)
            mini_disc = MinibatchDiscrimination(10, 3)(flat)

            md = Conv1D(8, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same')(signal)
            md = LeakyReLU(alpha=0.2)(md)
            md = Dropout(0.25)(md)
            md = MaxPooling1D(3)(md)

            md = Conv1D(16, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same')(md)
            md = LeakyReLU(alpha=0.2)(md)
            md = Dropout(0.25)(md)
            md = MaxPooling1D(3, strides=2)(md)

            md = Conv1D(32, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same')(md)
            md = LeakyReLU(alpha=0.2)(md)
            md = Dropout(0.25)(md)
            md = MaxPooling1D(3, strides=2)(md)

            md = Conv1D(64, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same')(md)
            md = LeakyReLU(alpha=0.2)(md)
            md = Dropout(0.25)(md)
            md = MaxPooling1D(3, strides=2)(md)
            md = Flatten()(md)
            concat = Concatenate()([md, mini_disc])
            validity = Dense(1, activation='sigmoid')(concat)

            return Model(inputs=signal, outputs=validity, name = "Discriminator")
            # return Model(inputs=signal, outputs=validity)



        else:
            model = Sequential(name='Discriminator')
            # model = Sequential()
            model.add(Conv1D(8, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(MaxPooling1D(3))

            model.add(Conv1D(16, kernel_size=8, strides=1, input_shape=self.input_shape, padding='same'))
            model.add(Dropout(0.25))
            model.add(MaxPooling1D(3, strides=2))

            model.add(Conv1D(32, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(MaxPooling1D(3, strides=2))

            model.add(Conv1D(64, kernel_size=8, strides=2, input_shape=self.input_shape, padding='same'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))
            model.add(MaxPooling1D(3, strides=2))

            model.add(Flatten())
            model.add(Dense(1, activation='sigmoid'))

            model.summary()

            validity = model(signal)

            return Model(inputs=signal, outputs=validity)
    
    def train(self, epochs, X_train, batch_size=128, save_interval=50, save=False, save_model_interval=1000):
        vaild = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))        
        
        for epoch in range(epochs):

            # -------------------
            # Train discriminator
            # -------------------

            # select a random batch of signals
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            signals = X_train[idx]

            # sample noise and generatir a batch of new signals
            noise = self.generate_noise(batch_size, self.random_sine)
            gen_signals = self.generator.predict(noise)
            
            # train the discriminator (real signals labeled as 1 and fake labeled as 0)
            d_loss_real = self.discrimintor.train_on_batch(signals, vaild)
            d_loss_fake = self.discrimintor.train_on_batch(gen_signals, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            #-------------------
            # Train Generator
            #-------------------
            
            # Train the generator (Goal: fool discriminator)
            g_loss = self.combine.train_on_batch(noise, vaild)

            # print the progresss
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # if reach save interval, plot the signals and save as image
            if epoch % save_interval == 0:
                self.save_image(epoch)
            if save:
                if os.path.isdir('save_model/') != True:
                    os.mkdir('save_model/')
                if (epoch % save_model_interval == 0 and epoch > 0):
                    self.generator.save('save_model/gen_%d.h5'%epoch)
                    self.disciminator.save('save_model/disc_%d.h5'%epoch)
            
        # save last round result
        self.save_image(epoch)
        self.generator.save('save_model/gen_%d.h5'%epoch)
        self.disciminator.save('save_model/disc_%d.h5'%epoch)

    def save_image(self, epoch):

        if os.path.isdir('image/') != True:
            os.mkdir('image/')

        r, c = 3, 3
        noise = self.generate_noise(r*c, self.random_sine)

        signals = self.generator.predict(noise) * self.scale

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

    def generate_noise(self, batch_size, sinwave=False):
        '''
        generate noise
        if sinwave is True, generate sin wave noise, otherwise, return standard normal distribution.
        '''
        if sinwave:
            x = np.linspace(-np.pi, np.pi, self.latent_size)
            noise = 0.1 * np.random.random_sample((batch_size, self.latent_size)) + 0.9 * np.sin(x)
        else:
            noise = np.random.normal(0, 1, size=(batch_size, self.latent_size))
        return noise

    def specify_range(self, signals, min_val=-1, max_val=1):
        """
        Specify acceptable range, drop signal if signal value is out of range.
        """

        if signals is None:
            raise ValueError("No signals data.")
        if type(signals) != np.ndarray :
            signals = np.array(signals)
        select_signals = []
        for sg in signals:
            min_sg = np.min(sg)
            max_sg = np.max(sg)

            if (min_sg >= min_val and max_sg <= max_val):
                select_signals.append(sg)
        
        return np.array(select_signals)


  
# if __name__ == "__main__":
#     EPOCHS = 3000
#     LATENT_SIZE = 50
#     SAVE_INTRIVAL = 100
#     BATCH_SIZE = 128
#     INPUT_SHAPE = (180, 1)
#     RANDOM_SINE = False

#     X_train = pickle.load(open('X_train.pkl', 'rb'))
#     dcgan = DCGAN(INPUT_SHAPE, LATENT_SIZE) 
#     X_train = dcgan.specify_range(X_train)
#     X_train = X_train.reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
#     dcgan.train(EPOCHS, X_train, BATCH_SIZE, SAVE_INTRIVAL)
#     print("Complete!!!")






