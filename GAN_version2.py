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
    lyr = layers.Conv1D(32, 16, activation=layers.LeakyReLU(alpha=0.2), padding='same')(input_ecg)
    lyr = layers.BatchNormalization()(lyr)
    lyr = layers.MaxPooling1D()(lyr)

    lyr = layers.Conv1D(64, 16, activation=layers.LeakyReLU(alpha=0.2), padding='same')(lyr)
    lyr = layers.BatchNormalization()(lyr)
    lyr = layers.MaxPooling1D()(lyr)

    lyr = layers.Conv1D(128, 16, activation=layers.LeakyReLU(alpha=0.2), padding='same')(lyr)
    lyr = layers.BatchNormalization()(lyr)
    lyr = layers.MaxPooling1D()(lyr)
    lyr = layers.Flatten()(lyr)
    # lyr = layers.GRU(30)(lyr)
    # lyr = layers.Dropout(0.5)(lyr)
    # lyr = layers.GaussianNoise(0.2)(lyr)
    # lyr = layers.Dense(n_class)(lyr)
    
    # label output
    lb_out = layers.Dense(n_class, activation='softmax')(lyr)
    # true/fake output
    tf_out = layers.Dense(1, activation='sigmoid')(lyr)
    d_model = tf.keras.Model(input_ecg, [tf_out, lb_out])
    # compile both model
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    d_model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    c_model = tf.keras.Model(input_ecg, lb_out)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])    
    return c_model, d_model
    

def define_generator(input_shape, n_class=4):
    in_latent = tf.keras.Input(shape=input_shape)
    n_nodes = 15*128

    in_label = layers.Input(shape=(1,))
    lb = layers.Embedding(n_class, 50)(in_label)
    lb = layers.Dense(15)(lb)
    lb = layers.Reshape((15, 1))(lb)
    gen = layers.Dense(n_nodes)(in_latent)
    gen = layers.LeakyReLU(alpha=0.2)(gen)
    gen = layers.Reshape((15, 128))(gen)

    gen = layers.Concatenate()([gen, lb])

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

    g_model = tf.keras.Model([in_latent, in_label], g_out)
    # model.summary()
    return g_model

def define_gan(d_model, g_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = tf.keras.Model(g_model.input, gan_output)
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt) 
    # model.summary()
    return model

def generate_latent(latent_size, n_sample, n_class=4):
    # generete latent point following normal distribution
    z_input = np.random.randn(latent_size * n_sample)
    # reshape data_size x latent_size
    z_input = np.reshape(z_input, (n_sample, latent_size))
    labels = np.random.randint(0, n_class, size=n_sample)

    return [z_input, labels.astype(np.float16)]
    
def generate_fake_sample(generator, latent_size, n_sample):
    z_input, labels = generate_latent(latent_size, n_sample)
    signals = generator.predict([z_input, labels])
    y = np.zeros((n_sample, 1))    #fake data y=0 --->fake:0, real:1
    return [signals, labels.astype(np.float16)], y.astype(np.float16)

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
    return [X, labels.astype(np.float16)], y.astype(np.float16)

def summarize_performance(step, g_model, c_model, latent_size, dataset, n_sample=100):

    [X, labels], _ = generate_fake_sample(g_model, latent_size, n_sample)

    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.axis('off')
        plt.plot(X[i, :, :].reshape(180,))
    filename = 'generated_plot_%04d.png' % (step + 1)
    plt.savefig(filename)
    plt.close()
    # X, y = dataset
    # _, acc = c_model.evaluate(X, labels.astype(np.float16), verbose=0)
    # print('Classifier Accuracy: %.3f%%' % (acc * 100))
    

def train(g_model, d_model, c_model, gan_model, dataset, latent_size, n_epochs=20, n_batch=100):
    # X_sup, y_sup = select_supervised_sample(dataset)
    X_sup, y_sup = dataset
    print(X_sup.shape, y_sup.shape)
    bat_per_epo = len(dataset[0])//n_batch
    n_step = int(bat_per_epo * n_epochs)
    half_batch = int(n_batch//2)
    #train classifier (c)
    # c_model.fit(X_sup, y_sup.astype(np.float16), batch_size=32, epochs=50)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_step))
    for i in range(n_step):
        
        # update unsupervised discriminator (d)
        [X_real, lb_real], y_real = generate_real_sample(dataset, half_batch)
        _, d_r1, d_r2 = d_model.train_on_batch(X_real, [y_real, lb_real])
        [X_fake, lb_fake], y_fake = generate_fake_sample(g_model, latent_size, half_batch)
        _, d_f1, d_f2 = d_model.train_on_batch(X_fake, [y_fake, lb_fake])
        # update generator (g)
        X_gan, lb_gan = generate_latent(latent_size, n_batch)
        y_gan = np.ones((n_batch,1)).astype(np.float16)
        _, g_1, g_2 = gan_model.train_on_batch([X_gan, lb_gan], [y_gan, lb_gan])
        # summarize loss on this batch
        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f1, d_f2, g_1,g_2))
        
        if (i + 1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, g_model, c_model, latent_size, dataset)
            

def load_data(path_train, path_label):
    ecg = np.array(ut.read_pickle(path_train))
    lb = ut.read_pickle(path_label)
    lb[lb=='N'] = 0
    lb[lb=='A'] = 1
    lb[lb=='O'] = 2
    lb[lb=='~'] = 3
    # lb = tf.keras.utils.to_categorical(lb, n_classes=4)
    return [ecg.reshape(-1, 180, 1), np.array(lb)]




if __name__ == "__main__":
    PATH_TRAIN = 'data/mw_train.pkl'
    PATH_LABEL = 'data/label_train.pk1'
    latent_size = 100
    c_model, d_model = define_discriminator((180, 1), 4)
    g_model = define_generator((latent_size, ), 4)
    gan_model = define_gan(d_model, g_model)
    dataset = load_data(PATH_TRAIN, PATH_LABEL)
    train(g_model, d_model, c_model, gan_model, dataset, latent_size)
    # d_model, c_model  = define_discriminator((180, 1))
    # d_model.summary()

    # c_model.summary()

