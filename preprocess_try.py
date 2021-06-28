import numpy as np
import biosppy as bio
from tqdm import tqdm
import os
import pickle
import pandas as pd
from scipy.io import loadmat


def load_data(path, save=False):
    ECG_dict = {}
    for r, d, f in os.walk(path):
        for file in tqdm(f):
            if '.mat' in file:
                name = file.replace('.mat', '')
                ECG_dict[name] = loadmat(os.path.join(r, file))['val'][0]/1000
    labels = np.array(pd.read_csv(os.path.join(path, 'REFERENCE.csv'), header=None))

    for i in tqdm(range(len(labels))):
        sg = labels[i,0]
        ECG_dict[sg] = [labels[i, 1], ECG_dict[sg]]

    if save:
        with open('ECG_data.pkl', 'wb') as f:
                pickle.dump(ECG_dict,f)
    return ECG_dict


def extract_heartbeats(signal, sampling_rate=300):
    if signal is None:
        raise TypeError("Please specify an input signal.")

    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    
    # filter signal
    order = int(0.3 * sampling_rate)
    filtered, _, _ = bio.signals.tools.filter_signal(signal=signal,
                                                     ftype='FIR',
                                                     band='bandpass',
                                                     order=order,
                                                     frequency=[3, 45],
                                                     sampling_rate=sampling_rate)
    
    # segment
    rpeaks,  = bio.signals.ecg.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)
    # correction r-peak
    rpeaks,  = bio.signals.ecg.correct_rpeaks(signal=filtered,
                                            rpeaks=rpeaks,
                                            sampling_rate=sampling_rate,
                                            tol=0.05)
    # extract templates
    heartbeats, rpeaks = bio.signals.ecg.extract_heartbeats(signal=filtered, 
                                                            rpeaks=rpeaks,
                                                            sampling_rate=sampling_rate,
                                                            before=0.2,
                                                            after=0.4)
    return (heartbeats, rpeaks, filtered)

def select_medwave(heartbeats):

    r_peaks = np.max(heartbeats, axis=1)
    med_val = np.sort(r_peaks)[len(r_peaks)//2]
    med_idx = list(r_peaks).index(med_val) #avoid mult value
    med_wave = heartbeats[med_idx]

    return med_wave


def process_signal(ecg_dataset, save=False):

    if type(ecg_dataset) != dict:
        raise TypeError('Wrong input data type, correct type: dict')

    median_wave = {}
    for i in tqdm(ecg_dataset.keys()):
        lb = ecg_dataset[i][0]
        sg = ecg_dataset[i][1]
        hb, rp, filt_sg = extract_heartbeats(signal=sg, sampling_rate=300)
        median_wave[i] = [lb,select_medwave(hb)]
    if save:
        with open('ECG_medwave.pkl', 'wb') as f:
            pickle.dump(median_wave,f)

    return median_wave

    
      







    

