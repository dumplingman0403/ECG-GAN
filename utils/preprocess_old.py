import numpy as np
import biosppy as bio
from tqdm import tqdm


SAMPLE_RATE = 300  # sample rate 300Hz


class PreprocessECG(object):

    def __init__(self):
        self.sample_rate = SAMPLE_RATE

    def run_preprocess(self, ECG_data):
        print("Start processing....\n")
        # Filtered_ECG = []
        Heartbeat_ECG = []
        # SelectHeartbeat_ECG = []
        for i, ecg in tqdm(enumerate(ECG_data)):
            # ts, filtered, rpeaks, templates_ts, templates, hr_rate_ts, hr_rate = bio.signals.ecg.ecg(
            #     ecg, self.sample_rate, show=False)
            _, _, _, templates_ts, templates, _, _ = bio.signals.ecg.ecg(
                ecg, self.sample_rate, show=False)
            Heartbeat_ECG.append(templates)
        print('Complete.\n')
        return Heartbeat_ECG

    def select_median_wave(self, heartbeat):
        rpeak = []
        for hb in heartbeat:
            rpeak.append(np.max(hb))
        
        rpeak_sort = sorted(rpeak.copy())
        med_rpeak = rpeak_sort[len(rpeak_sort)//2]
        med_idx = rpeak.index(med_rpeak)
        med_wave = heartbeat[med_idx]
        return med_wave

    def run_ecg_preprocess(self, ECG_data):
        heartbeat= self.run_preprocess(ECG_data)
        med_wave_list = []
        print('start selecting median wave...\n')
        for hb in tqdm(heartbeat):
            mv = self.select_median_wave(hb)
            med_wave_list.append(mv)
        print('complete.\n')
        return med_wave_list

if __name__ == "__main__":
    
    import pickle
    data = np.array(pickle.load(open('data/data_train.pk1', 'rb')))
    ecg_data = data[:, 1]
    prep_ecg = PreprocessECG()
    MedianWave = prep_ecg.run_ecg_preprocess(ecg_data)
    with open('mw_train.pkl', 'wb') as f:
        pickle.dump(MedianWave, f)


