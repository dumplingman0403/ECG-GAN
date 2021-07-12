# Synthesis ECG signals via Generative adversarial networks

- rnn.py  RNN classifier

## Installation
```bash
pip install -r requirement.txt
```
## Usage

### download dataset
- option 1
```
./download.sh
```
- download from website 
    - [AF Classification from a Short Single Lead ECG Recording - The PhysioNet Computing in Cardiology Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/training2017.zip)
    - [MIT-BIH Arrhythmia Database](https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip)

### process ECG signals
```
python3 process_ecg.py
```
## To do list
- [x] process ECG dataset - AF Classification Challenge 2017
- [x] process ECG dataset - MIT-BIH Arrhythmia Database
- [x] design GAN framework
- [ ] hyperparameter model 
- [ ] validation
