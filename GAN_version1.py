import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import backend as K
import utils as ut
from tqdm import tqdm
from matplotlib import pyplot

def define_discriminator(input_shape, n_class):

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

    # class label output  
    out1 = layers.Dense(n_class, activation='softmax')(lyr)
    # real/fake output
    out2 = layers.Dense(1, activation='sigmoid')(lyr)

    model = tf.keras.Model(input_ecg, [out1, out2])
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['sparse_categorical_crossentropy', 'binary_crossentropy'], optimizer=opt)
    
    model.summary()
        
    return model
    

def define_generator(input_shape, n_class=4):
    input_label = tf.keras.Input((1,))
    lyr_lb = layers.Embedding(n_class, 50)(input_label)
    n_nodes = 45*1
    lyr_lb = layers.Dense(n_nodes)(lyr_lb)
    lyr_lb = layers.Reshape((45, 1))(lyr_lb)



    input_latent = tf.keras.Input(input_shape)
    nodes = 45*12
    lyr = layers.Dense(nodes)(input_latent)
    lyr = layers.Reshape((45, 12))(lyr)
    lyr = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(lyr)
    merge = layers.Concatenate()([lyr, lyr_lb])
    lyr = layers.Conv1D(32, 3, activation='relu', padding='same')(merge)
    lyr = layers.UpSampling1D(size=2)(lyr)
    lyr = layers.Conv1D(16, 3, activation='relu', padding='same')(lyr)
    lyr = layers.UpSampling1D(size=2)(lyr)
    out_lyr = layers.Conv1D(1, 3, activation='tanh', padding='same')(lyr)

    model = tf.keras.Model([input_latent, input_label], out_lyr)
    # model.summary()
    return model

def generate_latent(latent_size, n_sample, num_class):
    # generete latent point following normal distribution
    latent_input = np.random.randn(latent_size * n_sample)
    # reshape data_size x latent_size
    latent_input = np.reshape(latent_input, (n_sample, latent_size))
    label = np.random.randint(0, num_class, n_sample)
    return [latent_input, label]
    
def generate_fake_sample(generator, latent_size, n_sample, n_class=4):
    latent_input, label = generate_latent(latent_size, n_sample, n_class)
    ecg = generator.predict([latent_input, label])
    y = np.zeros((n_sample, 1))    #fake data y=0 --->fake:0, real:1
    return [ecg, label], y

def generate_real_sample(dataset, n_sample):
    ecg, label = dataset
    #sample dataset
    idx= np.random.randint(0, len(dataset[0]), size=n_sample)
    sample_ecg, sample_label = ecg[idx], label[idx]
    sample_ecg = np.reshape(sample_ecg,(-1, 180, 1))
    sample_label = np.array(sample_label)
    y = np.ones((n_sample, 1))  #real data y=1 --->fake:0, real:1
    return [sample_ecg, sample_label], y

def define_gan(discriminator_model, generator_model):
    discriminator_model.trainable = False
    gan_output = discriminator_model(generator_model.output)
    model = tf.keras.Model(generator_model.input, gan_output)
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['sparse_categorical_crossentropy','binary_crossentropy'], optimizer=opt) 
    model.summary()
    return model

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100, n_class=4):
    [X_real, label_real], y_real = generate_real_sample(dataset, n_samples)
    [X_fake, label_fake], y_fake = generate_fake_sample(generator, latent_size, n_samples, n_class)
    _, acc_lr, acc_yr = d_model.evaluate(X_real.astype('float32'), [label_real.astype('float32'), y_real])
    _, acc_lf, acc_yf = d_model.evaluate(X_fake, [label_fake, y_fake])
    print('>Accuracy label= real: %.0f%%, fake: %.0f%% y= real: %.0f%%, fake: %.0f%%' % (acc_lr*100, acc_lf*100, acc_yr*100, acc_yf*100))

def train(discriminator_model, generator_model, gan_model, dataset, latent_size=100, n_epoch=30, n_batch=64, n_class=4):
    data_size = len(dataset[0])
    bat_per_epo = int(data_size/n_batch)
    print("batch per epoch: %d" % bat_per_epo)
    n_step = bat_per_epo * n_epoch
    print("number of steps: %d" % n_epoch)
    half_batch = int(n_batch/2)

    for i in range(n_epoch):
        for j in range(bat_per_epo):
            [X_real, label_real], y_real = generate_real_sample(dataset, half_batch)
            [X_fake, label_fake], y_fake = generate_fake_sample(generator_model, latent_size, half_batch, n_class)

            d_r= discriminator_model.train_on_batch(X_real.astype('float32'), [label_real.astype('float32'), y_real]) 
            d_f= discriminator_model.train_on_batch(X_fake, [label_fake, y_fake])

            [z_input, z_labels] = generate_latent(latent_size, n_batch, 4)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])
            print('>%d, %d/%d' % (i+1, j+1, bat_per_epo), d_r, d_f, g_loss)
        if (i+1) % 5 == 0:
            summarize_performance(i, generator_model, discriminator_model, dataset, latent_size)

def load_data():
    ecg = np.array(ut.read_pickle('data/MedianWave_train.pk1'))
    lb = ut.read_pickle('data/label_train.pk1')
    lb[lb=='N'] = 0
    lb[lb=='A'] = 1
    lb[lb=='O'] = 2
    lb[lb=='~'] = 3
    # lb = tf.keras.utils.to_categorical(lb, num_classes=4)
    return [ecg, np.array(lb)]




if __name__ == "__main__":
    
    latent_size = 100
    discriminator = define_discriminator((180, 1), 4)
    generator = define_generator((latent_size, ), 4)
    gan_model = define_gan(discriminator, generator)
    dataset = load_data()
    # print(len(dataset[0]))
    train(discriminator, generator, gan_model, dataset, latent_size)