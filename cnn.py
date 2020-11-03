import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import pickle
import numpy as np


def cnn_model(input_size):

    model = tf.keras.Sequential()
    model.add(layers.Conv1D(16, 3, activation='relu', input_shape=input_size))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    model.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def PrepareInput(train_path, val_path, train_label, val_label):

    train_data = pickle.load(open(train_path, 'rb'))
    val_data = pickle.load(open(val_path, 'rb'))
    train_label = pickle.load(open(train_label, 'rb'))
    val_label = pickle.load(open(val_label, 'rb'))

    x_train = np.reshape(train_data, (-1, 180, 1))
    x_val = np.reshape(val_data, (-1, 180, 1))

    y_train = np.array(train_label)
    y_train[y_train == 'N'] = 0
    y_train[y_train == 'A'] = 1
    y_train[y_train == 'O'] = 2
    y_train[y_train == '~'] = 3

    y_val = np.array(val_label)
    y_val[y_val == 'N'] = 0
    y_val[y_val == 'A'] = 1
    y_val[y_val == 'O'] = 2
    y_val[y_val == '~'] = 3

    y_train = tf.keras.utils.to_categorical(y_train, 4)
    y_val = tf.keras.utils.to_categorical(y_val, 4)

    return [x_train, x_val, y_train, y_val]
    model = cnn_model()

def Run(epochs):
    PATH_TRAIN = 'data/mw_train.pkl'
    PATH_VAL = 'data/mw_test.pkl'
    TRAIN_LABEL = 'data/label_train.pk1'
    VAL_LABEL = 'data/label_sample.pk1'
    x_train, x_val, y_train, y_val = PrepareInput(
        PATH_TRAIN, PATH_VAL, TRAIN_LABEL, VAL_LABEL)
    Cnn = cnn_model(x_train.shape[1:])
    history = Cnn.fit(x_train, y_train, epochs=epochs, verbose=1,
                      batch_size=32, validation_data=(x_val, y_val))
    return history
    
if __name__ == "__main__":
    history = Run(50)
