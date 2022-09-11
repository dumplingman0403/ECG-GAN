# Synthesis ECG signals via Generative adversarial networks


## Setup
```bash
pip install -r requirement.txt
```
## Usage
### [See data description](utils/README.md)
### download dataset

- option 1
```
sh download.sh
```
if you haven't install `unzip`, install `unzip` package first before run `download.sh`<br>
install guildline: <br>
[General](https://www.tecmint.com/install-zip-and-unzip-in-linux/) - for Linux<br>
[Homebrew](https://formulae.brew.sh/formula/unzip) - for MacOS 
- option 2:
    download from website 
    - [AF Classification from a Short Single Lead ECG Recording - The PhysioNet Computing in Cardiology Challenge 2017](https://physionet.org/content/challenge-2017/1.0.0/training2017.zip)
    - [MIT-BIH Arrhythmia Database](https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip)

### Process ECG signals
run following command 
```
python3 process_ecg.py
```
if you change the dataset path, modify `process_ecg.py` 
```python 3
# modify dataset path if necessary
AA_DATASET_DIR = 'AA_dataset/'   # MIT-BIH Arrhythmia Database
    
AF_DATASET_DIR = 'AF_dataset/'   # AF Classification from a Short Single Lead ECG Recording - The PhysioNet Computing in Cardiology Challenge 2017
LABEL_PATH = 'AF_dataset/REFERENCE-original.csv'

``` 
### GAN Training
training ECG signal with GAN model
```
python3 train.py
```

modify the input dataset path in `train.py` if you change the path of `X_train_af.pkl` and `y_af.pkl`
```
X_train = pickle.load(open("path_of_X_train_af.pkl", "rb"))  # --> load AF dataset
y = pickle.load(open("path_of_y_af.pkl", 'rb'))
```

## Output 
### MIT-BIH Arrhythmia Database
<img src="generate_ECG/aa_e4000_7.png" alt="aa_e4000_7.png" width="150"/><img src="generate_ECG/aa_e4000_16.png" alt="aa_e4000_16.png" width="150"/><img src="generate_ECG/aa_e4000_11.png" alt="aa_e4000_11.png" width="150"/><img src="generate_ECG/aa_e4000_19.png" alt="aa_e4000_19.png" width="150"/><img src="generate_ECG/aa_e4000_40.png" alt="aa_e4000_40.png" width="150"/><img src="generate_ECG/aa_e5000_12.png" alt="aa_e5000_12.png" width="150"/><img src="generate_ECG/aa_e5000_26.png" alt="aa_e5000_26.png" width="150"/><img src="generate_ECG/aa_e7000_44.png" alt="aa_e7000_44.png" width="150"/><img src="generate_ECG/aa_e8000_33.png" alt="aa_e8000_33.png" width="150"/><img src="generate_ECG/aa_e10000_51.png" alt="aa_e10000_51.png" width="150"/><img src="generate_ECG/aa_e10000_89.png" alt="aa_e10000_89.png" width="150"/>

### Short Single Lead ECG Recording
#### Atrial Fibrillation
real <br>
<img src="generate_ECG/af_real_3.png" alt="af_real_3.png" width="150"/>
<img src="generate_ECG/af_real_4.png" alt="af_real_4.png" width="150"/>
<img src="generate_ECG/af_real_6.png" alt="af_real_6.png" width="150"/> <br>
epoch 1000 <br>
<img src="generate_ECG/afaf_e1000_3.png" alt="afaf_e1000_3.png" width="150"/>
<img src="generate_ECG/afaf_e1000_4.png" alt="afaf_e1000_4.png" width="150"/>
<img src="generate_ECG/afaf_e1000_6.png" alt="afaf_e1000_6.png" width="150"/> <br>
epoch 2000 <br>
<img src="generate_ECG/afaf_e2000_6.png" alt="afaf_e2000_6.png" width="150"/>
<img src="generate_ECG/afaf_e2000_8.png" alt="afaf_e2000_8.png" width="150"/>
<img src="generate_ECG/afaf_e2000_14.png" alt="afaf_e2000_14.png" width="150"/> <br>