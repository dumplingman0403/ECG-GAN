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
    lyr = layers.Flatten()(lyr)
    # lyr = layers.GRU(30)(lyr)
    # lyr = layers.Dropout(0.5)(lyr)
    # lyr = layers.GaussianNoise(0.2)(lyr)

    # class label output  
    out1 = layers.Dense(n_class, activation='softmax')(lyr)
    # real/fake output
    out2 = layers.Dense(1, activation='sigmoid')(lyr)

    model = tf.keras.Model(input_ecg, [out1, out2])
    opt = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=[tf.keras.losses.sparse_categorical_crossentropy, 'binary_crossentropy'], optimizer=opt)
    
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
    model.compile(loss=[tf.keras.losses.sparse_categorical_crossentropy,'binary_crossentropy'], optimizer=opt) 
    model.summary()
    return model

def train(discriminator_model, generator_model, gan_model, dataset, latent_size=100, n_epoch=30, n_batch=64, n_class=4):
    data_size = len(dataset[0])
    bat_per_epo = int(data_size/n_batch)
    print("batch per epoch: %d" % bat_per_epo)
    n_step = bat_per_epo * n_epoch
    print("number of steps: %d" % n_epoch)
    half_batch = int(n_batch/2)

    for i in range(n_step):

        [X_real, label_real], y_real = generate_real_sample(dataset, half_batch)   #sample size = half batch???

        _, d_r1, d_r2 = discriminator_model.train_on_batch(X_real.astype('float32'), [label_real.astype('float32'), y_real])

        [X_fake, label_fake], y_fake = generate_fake_sample(generator, latent_size, half_batch, n_class)

        _, d_f1, d_f2 = discriminator_model.train_on_batch(X_fake, [label_fake, y_fake])
        
        [z_input, z_labels] = generate_latent(latent_size, n_batch, 4)

        y_gan = np.ones((n_batch, 1))

        _, g_1, g_2 = gan_model.train_on_batch([z_input, z_labels], [y_gan, z_labels])

        print('>%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % (i+1, d_r1,d_r2, d_f1,d_f2, g_1,g_2))

        if (i+1) % (bat_per_epo * 1) == 0:
            summarize_performance(i, generator_model, latent_size)
def load_data():
    ecg = np.array(ut.read_pickle('data/MedianWave_train.pk1'))
    lb = ut.read_pickle('data/label_train.pk1')
    lb[lb=='N'] = 0
    lb[lb=='A'] = 1
    lb[lb=='O'] = 2
    lb[lb=='~'] = 3
    # lb = tf.keras.utils.to_categorical(lb, num_classes=4)
    return [ecg, np.array(lb)]



def summarize_performance(step, g_model, latent_dim, n_samples=100):
    # prepare fake examples
    [X, nmn_label], nmn_y = generate_fake_sample(g_model, latent_dim, n_samples) #TODO!:Numan (nmns were _ and _) - change labels in this row and debug!
    nmn_label.reshape((1, len(nmn_label)))
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # plot images
    for i in range(100):
        # define subplot
        pyplot.subplot(10, 10, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(X[i, :], cmap='gray_r')
        # np.savetxt('test_raw_nc%d%d.csv' % (i,step), X[i,:], delimiter=',')
        # np.savetxt('test_cat_nc%d%d.csv' % (i,step), nmn_label[i],delimiter=',')
    # save plot to file
    #np.savetxt('test_raw_nc%d.csv' % (step), X[:,:,0], delimiter=',')
    #np.savetxt('test_cat_nc%d.csv' % (step), nmn_label[:],delimiter=',')
    filename1 = 'generated_plot_%04d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = 'model_%04d.h5' % (step+1)
    g_model.save(filename2)
    print('>Saved: %s and %s' % (filename1, filename2))
if __name__ == "__main__":
    
    latent_size = 100
    discriminator = define_discriminator((180, 1), 4)
    generator = define_generator((latent_size, ), 4)
    gan_model = define_gan(discriminator, generator)
    dataset = load_data()
    # print(len(dataset[0]))
    train(discriminator, generator, gan_model, dataset, latent_size)