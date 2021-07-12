import numpy as np
import wfdb
import os
import pickle
import biosppy as bio
from tqdm import tqdm

def load_from_path(path, save=False):

    file_list = os.listdir(path)
    ECG = {}
    for file in file_list:

        if '.dat' in file:
            file_name = file.replace('.dat', '')
            record = wfdb.io.rdsamp(os.path.join(path,file_name))
            
            ECG[file_name] = record[0][:, 0]
    if save:
        with open("arrhyth_dataset.pkl", "wb") as f:
            pickle.dump(ECG, f)

    return ECG

def extract_heartbeats(signal, sampling_rate):
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

def process_signals(signals, sampling_rate, save=False):
        
    if signals is None:
        raise TypeError("Please specify input signals.")
    print('Start processing ECG signals...')
    ECG_heartbeats = {}
    if type(signals) != dict:
        raise TypeError('Input type error, signals must be dictionary.')

    for sg_id in tqdm(signals.keys()):
        sg = signals[sg_id]
        # lb = str(signals[sg_id][0])
        heartbeats, rpeaks, filtered = extract_heartbeats(sg, sampling_rate=sampling_rate)
        ECG_heartbeats[sg_id] = heartbeats
    
    if save:
        with open('arrhythmia_heartbeats.pkl', 'wb') as f:
            pickle.dump(ECG_heartbeats, f)
    
    return ECG_heartbeats

if __name__ == '__main__':

    PATH = "mit-bih-arrhythmia-database-1.0.0" 

    ECG_data = load_from_path(PATH, save=True)

    Heartbeats = process_signals(ECG_data, 360, True)









