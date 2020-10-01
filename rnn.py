import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import pickle
import numpy as np


data = np.array(pickle.load(open('filtered_sample.pk1', 'rb')))
label = pickle.load(open('label_sample.pk1', 'rb'))
label = np.array(label)
label = pickle.load(open('label_sample.pk1', 'rb'))
label = np.array(label)
label[label == 'N'] = 0
label[label == 'A'] = 1
label[label == 'O'] = 2
label[label == '~'] = 3
label = tf.keras.utils.to_categorical(label, 4)
ECG_list = []
Y = []
for i, ecg in enumerate(data):
    if len(ecg) == 9000:
        ECG_list.append(list(ecg))
        Y.append(label[i])
Y = np.array(Y)
ECG_list = np.reshape(ECG_list, (-1, 9000, 1))
model = tf.keras.models.Sequential([tf.keras.Input(shape = (9000, 1))])
model.add(layers.LSTM(20))
model.add(layers.Dense(10))
model.add(layers.Dense(4, activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(ECG_list, Y, batch_size = 32, epochs = 10, validation_split = 0.1)



