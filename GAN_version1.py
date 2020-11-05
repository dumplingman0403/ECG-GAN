import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import backend as K
import utils as ut
from tqdm import tqdm
import matplotlib.pyplot as plt

def custom_activation(in_val):
    logexpsum = K.sum(K.exp(in_val), axis=-1, keepdims=True)
    result = logexpsum/(logexpsum + 1.0)
    return result

def define_discriminator(input_shape, n_class=4):

    input_ecg = tf.keras.Input(input_shape)
    lyr = layers.Conv1D(8, 3, activation=layers.LeakyReLU(alpha=0.2), padding='same')(input_ecg)
    lyr = layers.BatchNormalization()(lyr)
    lyr = layers.MaxPooling1D()(lyr)

    lyr = layers.Conv1D(16, 3, activation=layers.LeakyReLU(alpha=0.2), padding='same')(lyr)
    lyr = layers.BatchNormalization()(lyr)
    lyr = layers.MaxPooling1D()(lyr)

    lyr = layers.Conv1D(32, 3, activation=layers.LeakyReLU(alpha=0.2), padding='same')(lyr)
    lyr = layers.BatchNormalization()(lyr)
    lyr = layers.MaxPooling1D()(lyr)
    # lyr = layers.Flatten()(lyr)
    lyr = layers.GRU(30)(lyr)
    lyr = layers.Dropout(0.5)(lyr)
    lyr = layers.GaussianNoise(0.2)(lyr)
    lyr = layers.Dense(n_class)(lyr)
    # supervised output
    c_out = layers.Activation('softmax')(lyr)  
    c_model = tf.keras.Model(input_ecg, c_out)
    # unspervised output
    d_out = layers.Lambda(custom_activation)(lyr)
    d_model = tf.keras.Model(input_ecg, d_out)
    # compile both model
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt)
    d_model.compile(loss='binary_crossentropy', optimizer=opt)
        
    return c_model, d_model
    

def define_generator(input_shape, n_class=4):
    in_latent = tf.keras.Input(shape=input_shape)
    n_nodes = 15*128
    gen = layers.Dense(n_nodes)(in_latent)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Reshape((15, 128))(gen)

    gen = layers.Conv1D(128, 16, padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.UpSampling1D(2)(gen)
    gen = layers.BatchNormalization()(gen)

    gen = layers.Conv1D(64, 16, padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.UpSampling1D(3)(gen)
    gen = layers.BatchNormalization()(gen)

    gen = layers.Conv1D(32, 16, padding='same')(gen)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.UpSampling1D(2)(gen)
    gen = layers.BatchNormalization()(gen)

    g_out = layers.Conv1D(1, 16, activation='tanh',padding='same')(gen)
    # g_out = layers.Permute((2,1))(gen)

    g_model = tf.keras.Model(in_latent, g_out)
    # model.summary()
    return g_model

def define_gan(d_model, g_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = tf.keras.Model(g_model.input, gan_output)
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy'], optimizer=opt) 
    # model.summary()
    return model

def generate_latent(latent_size, n_sample):
    # generete latent point following normal distribution
    z_input = np.random.randn(latent_size * n_sample)
    # reshape data_size x latent_size
    z_input = np.reshape(z_input, (n_sample, latent_size))

    return z_input
    
def generate_fake_sample(generator, latent_size, n_sample):
    z_input = generate_latent(latent_size, n_sample)
    signals = generator.predict(z_input)
    y = np.zeros((n_sample, 1))    #fake data y=0 --->fake:0, real:1
    return signals, y

def select_supervised_sample(dataset, n_sample=100, n_class=4):
    X, y = dataset
    X_list, y_list = list(), list()
    n_per_class = n_sample // n_class

    for i in range(n_class):
        X_with_class = X[y == 1]
        idx = np.random.randint(0, len(X_with_class), n_per_class)
        [X_list.append(X_with_class[j]) for j in idx]
        [y_list.append(i) for k in idx]
    return np.asarray(X_list).reshape(-1, len(X_list[0]), 1), np.asarray(y_list)

def generate_real_sample(dataset, n_sample):

    X, labels = dataset

    idx = np.random.randint(0, len(X), n_sample)
    X, labels = X[idx], labels[idx]
    X.reshape(-1,X.shape[1], 1)
    y = np.ones((n_sample, 1))
    return [X, labels], y

def summarize_performance(step, g_model, c_model, latent_size, dataset, n_sample=100):

    X, _ = generate_fake_sample(g_model, latent_size, n_sample)

    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.imshow(X[i, :, :])
    X, y = dataset
    _, acc = c_model.evaluate((X, y), verbose=0)
    print('Classifier Accuracy: %.3f%%' % (acc * 100))
    

def train(g_model, d_model, c_model, gan_model, dataset, latent_size, n_epochs=20, n_batch=100):
    X_sup, y_sup = select_supervised_sample(dataset)
    print(X_sup.shape, y_sup.shape)
    bat_per_epo = len(dataset[0])//n_batch
    n_step = int(bat_per_epo * n_epochs)
    half_batch = int(n_batch//2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_step))
    for i in range(n_step):
        # update supervised discriminator (c)
        [X_sup_real, y_sup_real], _ = generate_real_sample([X_sup, y_sup], half_batch)
        c_loss = c_model.train_on_batch(X_sup_real, y_sup_real)
        # update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_sample(dataset, half_batch)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_sample(g_model, latent_size, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent(latent_size, n_batch), np.ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        print('>%d, c[%.3f], d[%.3f,%.3f], g[%.3f]'%(i+1, c_loss, d_loss1, d_loss2, g_loss))
        
        if (i + 1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, c_model, latent_size, dataset)
            


# def train(d_model, g_model, gan_model, dataset, latent_size=100, n_epoch=30, n_batch=64, n_class=4):
#     data_size = len(dataset[0])
#     bat_per_epo = (data_size//n_batch)
#     print("batch per epoch: %d" % bat_per_epo)
#     n_step = bat_per_epo * n_epoch
#     print("number of steps: %d" % n_epoch)
#     half_batch = n_batch//2

#     for i in range(n_epoch):
        
#         [X_real, label_real], y_real = generate_real_sample(dataset, half_batch)
#         [X_fake, label_fake], y_fake = generate_fake_sample(g_model, latent_size, half_batch, n_class)

#         d_r= d_model.train_on_batch(X_real.astype('float32'), [label_real.astype('float32'), y_real]) 
#         d_f= d_model.train_on_batch(X_fake, [label_fake, y_fake])

#         [z_input, z_labels] = generate_latent(latent_size, n_batch, 4)
#         y_gan = np.ones((n_batch, 1))
#         g_loss = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
#         print('>%d, %d/%d' % (i+1, j+1, bat_per_epo), d_r, d_f, g_loss)
#         if (i+1) % 5 == 0:
#             summarize_performance(i, g_model, d_model, dataset, latent_size)

def load_data():
    ecg = np.array(ut.read_pickle('data/mw_train.pkl'))
    lb = ut.read_pickle('data/label_train.pk1')
    lb[lb=='N'] = 0
    lb[lb=='A'] = 1
    lb[lb=='O'] = 2
    lb[lb=='~'] = 3
    # lb = tf.keras.utils.to_categorical(lb, n_classes=4)
    return [ecg.reshape(-1, 180, 1), np.array(lb)]




if __name__ == "__main__":
    
    latent_size = 100
    c_model, d_model = define_discriminator((180, 1), 4)
    g_model = define_generator((latent_size, ), 4)
    gan_model = define_gan(d_model, g_model)
    dataset = load_data()
    # # print(len(dataset[0]))
    train(g_model, d_model, c_model, gan_model, dataset, latent_size)

