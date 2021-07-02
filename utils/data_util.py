from scipy.io import loadmat
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import pickle
import biosppy as bio

class DataLoader():

    def __init__(self, path):
        pass


    def load_data(self, data_path, label_path, save=False):
        ECG_dict = {}
        for r, d, f in os.walk(data_path):  #root, directory, files
            for file in tqdm(f):
                if '.mat' in file:
                    name = file.replace('.mat', '')
                    ECG_dict[name] = loadmat(os.path.join(r, file))['val'][0]/1000
        labels = np.array(pd.read_csv(label_path, header=None))

        for i in tqdm(range(len(labels))):
            sg = labels[i, 0]
            ECG[sg] = [labels[i, 1], ECG_dict[sg]]

        if save:
            with open('ECG_data.pkl', 'wb') as f:
                pickle.dump(ECG_dict, f)
        
        return ECG_dict
        
    def extract_heartbeats(self, signal, sampling_rate):
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
        
        # segmentation, detect R peaks
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

    def select_medwave(self, heartbeats):

        r_peaks = np.max(heartbeats, axis=1)
        med_val = np.sort(r_peaks)[len(r_peaks)//2]
        med_idx = list(r_peaks).index(med_val) #avoid mult value
        med_wave = heartbeats[med_idx]

        return med_wave

    def process_signals(self, signals, save=False):

        ECG_heartbeats = {}
        if type(signals) != dict:
            raise TypeError('Input type error, signals must be dictionary.')
        for sg_id in signals.keys():
            sg = signals[sg_id][1]
            lb = str(signals[sg_id][0])
            heartbeats, rpeaks, filtered = self.extract_heartbeats(sg)
            ECG_heartbeats[sg_id] = [lb, heartbeats]
        
        if save:
            with open('ECG_heartbeats.pkl', 'wb') as f:
                pickle.dump(ECG_heartbeats, f)
        
        return ECG_heartbeats