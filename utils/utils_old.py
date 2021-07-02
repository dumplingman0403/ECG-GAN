import os
import pickle
from scipy.io import loadmat
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_data(path):
    name_list = pd.read_csv(os.path.join(path, 'REFERENCE.CSV'), header=None)
    file_list = os.listdir(path)
    file_list = sorted(file_list)
    ECG_records = []
    ECG_name = np.array(name_list.iloc[:, 0])
    ECG_type = np.array(name_list.iloc[:, 1])
    for i, f in tqdm(enumerate(ECG_name)):
        filename = f + '.mat'
        file_path = os.path.join(path, filename)
        ecg = loadmat(file_path)
        ecg = np.array(ecg['val'][0]/1000)
        ecg_type = ECG_type[i]

        ECG_records.append([f, ecg, ecg_type])

    return ECG_records


def save_as_pickle(var, save_dir):
    save_dir = str(save_dir)
    with open(save_dir, 'wb') as f:
        pickle.dump(var, f)


def read_pickle(path):
    data = pickle.load(open(path, 'rb'))
    return data


def normalize(array):  # normalize range [-1, 1]
    array = np.array(array)
    max_val = np.max(array)
    min_val = np.min(array)
    norm_array = 2*((array - min_val)/(max_val - min_val)) - 1
    return np.around(norm_array, decimals=3)


if __name__ == "__main__":

    PATH_train = "/Users/ericwu/Documents/dataset/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0/training2017"
    PATH_val = "/Users/ericwu/Documents/dataset/af-classification-from-a-short-single-lead-ecg-recording-the-physionet-computing-in-cardiology-challenge-2017-1.0.0/sample2017/validation"

    data_train = load_data(PATH_train)
    data_val = load_data(PATH_val)

    save_as_pickle(data_train, "data_train.pk1")
    save_as_pickle(data_val, "data_val.pk1")
