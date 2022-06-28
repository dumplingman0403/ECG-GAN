from utils.data_util import *
import pickle


if __name__ == '__main__':

    # modify dataset path if necessary
    AA_DATASET_DIR = 'AA_dataset/'   # MIT-BIH Arrhythmia Database
    
    AF_DATASET_DIR = 'AF_dataset/'   # AF Classification from a Short Single Lead ECG Recording - The PhysioNet Computing in Cardiology Challenge 2017
    LABEL_PATH = 'AF_dataset/REFERENCE-original.csv'


    dataloader = DataLoader()
    print("Loading Data ...")
    ECG_AF = dataloader.load_af_challenge_db(AF_DATASET_DIR, LABEL_PATH, save=True)
    print("Processing ECG signal...")
    print("Processing AF dataset...")
    AF_hrbt = dataloader.process_signals(signals=ECG_AF, sampling_rate=300, save=True, save_name='AF_heartbeat.pkl')
    X_af, y_af = dataloader.prepare_input_challenge(AF_hrbt, save=True)
    print("Processing AA dataset...")
    ECG_AA = dataloader.load_arrhythmia_DB(AA_DATASET_DIR, save=True)
    AA_hrbt = dataloader.process_signals(ECG_AA, sampling_rate=360, save=True, save_name='AA_heartbeats.pkl', aa_data=True)
    # AA_hrbt = pickle.load(open('AA_heartbeats.pkl', 'rb'))
    X_aa, y_aa = dataloader.prepare_input_arrhythmia(AA_hrbt, save=True)
