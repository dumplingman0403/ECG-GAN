from scipy.io import loadmat
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import pickle
import biosppy as bio
from sklearn.preprocessing import MinMaxScaler
import wfdb

class DataLoader():



    def load_af_challenge_db(self, data_path, label_path, save=False, save_name='ECG_data.pkl'):
        print('Load data from directory...')
        ECG_dict = {}
        for r, d, f in os.walk(data_path):  #root, directory, files
            for file in tqdm(f):
                if '.mat' in file:
                    name = file.replace('.mat', '')
                    ECG_dict[name] = loadmat(os.path.join(r, file))['val'][0]/1000
        labels = np.array(pd.read_csv(label_path, header=None))

        for i in tqdm(range(len(labels))):
            sg = labels[i, 0]
            ECG_dict[sg] = [labels[i, 1], ECG_dict[sg]]

        if save:
            with open(save_name, 'wb') as f:
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

        if heartbeats is None:
            raise ValueError("Please specify input heartbeats.")

        r_peaks = np.max(heartbeats, axis=1)
        med_val = np.sort(r_peaks)[len(r_peaks)//2]
        med_idx = list(r_peaks).index(med_val) #avoid mult value
        med_wave = heartbeats[med_idx]

        return med_wave

    def process_signals(self, signals, sampling_rate, save=False, save_name='ECG_heartbeats.pkl', aa_data = False):
        
        if signals is None:
            raise TypeError("Please specify input signals.")
        print('Start processing ECG signals...')
        ECG_heartbeats = {}
        if type(signals) != dict:
            raise TypeError('Input type error, signals must be dictionary.')
        for sg_id in tqdm(signals.keys()):
            
            if aa_data:
                sg = signals[sg_id]
            else:
                sg = signals[sg_id][1]
                lb = str(signals[sg_id][0])
            heartbeats, rpeaks, filtered = self.extract_heartbeats(sg, sampling_rate=sampling_rate)

            if aa_data:
                ECG_heartbeats[sg_id] = heartbeats
            else:
                ECG_heartbeats[sg_id] = [lb, heartbeats]
        
        if save:
            with open(save_name, 'wb') as f:
                pickle.dump(ECG_heartbeats, f)
        
        return ECG_heartbeats


    def prepare_input_challenge(self, dataset, save=False, normalize=False, save_name = ('X_train_af.pkl', 'y_af.pkl')):

        if type(dataset) != dict:
            raise TypeError('Dateset type must be dictionary.')
        X_train = []
        y = []
        for sg_id in dataset.keys():
            lb = str(dataset[sg_id][0])
            if lb != '~':    # discard noise ECG which can't be identify 
                signal = dataset[sg_id][1]
                for hb in signal:

                    if normalize:
                        hb, _ = self.normalize(hb)

                    X_train.append(hb)
                    y.append(lb)
        
        X_train = np.round(np.array(X_train), 4)
        y = np.array(y)
        y[y=='N'] = 0   # label normal ECG to 0 
        y[y=='A'] = 1   # label af ECG to 1 
        y[y=='O'] = 2   # label other ECG to 2
        y = y.astype(int)
         
        if save:
            with open(save_name[0], 'wb') as f:
                pickle.dump(X_train, f)
            with open(save_name[1], 'wb') as f:
                pickle.dump(y, f)
        
        return (X_train, y)

    def scale_signal(self, signals):
        """
        rescale signals
        signals: np.array
        """
        if type(signals) != np.ndarray:
            signal = np.array(signals)
        
        max_val = np.max(signals)
        min_val = np.min(signals)
        scale = max_val - min_val
        scale_signals = signals/scale
        return (scale_signals, scale)

    def specify_range(self, signals, min_val=-1, max_val=1):
        """
        Specify acceptable range, drop signal if signal value is out of range.
        """

        if not signals:
            raise ValueError("No signals data.")
        if type(signals) != np.ndarray :
            signals = np.array(signals)
        select_signals = []
        for sg in signals:
            min_sg = np.min(sg)
            max_sg = np.max(sg)

            if (min_sg >= -1 and max_val <= 1):
                select_signals.append(sg)
        
        return np.array(select_signals)

    def load_arrhythmia_DB(self, path, save=False):

        file_list = os.listdir(path)
        ECG = {}
        print('Load data from directory...')
        for file in tqdm(file_list):

            if '.dat' in file:
                file_name = file.replace('.dat', '')
                record = wfdb.io.rdsamp(os.path.join(path,file_name))
                
                ECG[file_name] = record[0][:, 0]
        if save:
            with open("arrhyth_dataset.pkl", "wb") as f:
                pickle.dump(ECG, f)

        return ECG

    def prepare_input_arrhythmia(self, dataset, save=False, save_name=('X_train_aa.pkl', 'y_aa.pkl')):
        if type(dataset) != dict:
            raise TypeError('Dateset type must be dictionary.')
        
        X_train = []
        y = []
        for sg_id in dataset.keys():

            sg = dataset[sg_id]
            for signal in sg:
                X_train.append(signal)
                y.append(sg_id)
        
        X_train = np.round(np.array(X_train), 4)
        y = np.array(y).astype(int)

        if save:
            with open(save_name[0], 'wb') as f:
                pickle.dump(X_train, f)
            with open(save_name[1], 'wb') as f:
                pickle.dump(y, f)

        return (X_train, y)

    def pick_type_only(self, X, y, target):
        """
        X: numpy array, training data
        y: numpy array, label
        target: int, target type of signal
        """
        pick_signals = []
        for i, type in enumerate(y):

            if type == target:
                pick_signals.append(X[i])

        return np.array(pick_signals)



