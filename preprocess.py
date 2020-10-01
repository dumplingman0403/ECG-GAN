import numpy as np
import biosppy as bio
from tqdm import tqdm


SAMPLE_RATE = 300  # sample rate 300Hz


class PreprocessECG(object):

    def __init__(self):
        self.sample_rate = SAMPLE_RATE

    def run_preprocess(self, ECG_data):
        print("Start processing....")
        Filtered_ECG = []
        Heartbeat_ECG = []
        SelectHeartbeat_ECG = []
        for i, ecg in tqdm(enumerate(ECG_data)):
            ts, filtered, rpeaks, templates_ts, templates, hr_rate_ts, hr_rate = bio.signals.ecg.ecg(
                ecg, self.sample_rate, show=False)

            Filtered_ECG.append(filtered)

            select_heartbeat = self.select_heartbeat(templates)

            SelectHeartbeat_ECG.append(select_heartbeat)
            Filtered_ECG.append(filtered)
            Heartbeat_ECG.append(templates)

        print('Complete')
        return [Filtered_ECG, Heartbeat_ECG, SelectHeartbeat_ECG]

    def select_heartbeat(self, heartbeat, select_num=6):
        heartbeat = np.array(heartbeat)
        random_indice = np.random.choice(
            len(heartbeat), size=select_num, replace=False)
        select_hrtbeat = heartbeat[random_indice, :]
        return select_hrtbeat


def select_MedWave(heartbeat):

    median_wave = []

    for hr in heartbeat:
        rpeak_list = []
        for sg in hr:
            rpeak = max(sg)
            rpeak_list.append(rpeak)

        rpeak_list_copy = rpeak_list.copy()
        rpeak_list_copy.sort()
        median_rpeak = rpeak_list_copy[len(rpeak_list_copy)//2]
        select_idx = rpeak_list.index(median_rpeak)
        mw = hr[select_idx]
        median_wave.append(mw)

    return median_wave


"""
test
"""
if __name__ == "__main__":
    import pickle
    data = np.array(pickle.load(open('data_val.pk1', 'rb')))
    ecg_data = data[:, 1]
    prep = PreprocessECG()
    Filtered_ECG, Heartbeat_ECG, SelectHeartbeat_ECG = prep.run_preprocess(
        ecg_data)
    with open('Heartbeat_val.pk1', 'wb') as f:
        pickle.dump(Heartbeat_ECG, f)
    # with open('label_train.pk1', 'wb') as f:
    #     pickle.dump(data[:, 2], f)
