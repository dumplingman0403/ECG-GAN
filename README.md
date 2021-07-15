# Synthesis ECG signals via Generative adversarial networks


## Setup
```bash
pip install -r requirement.txt
```
## Usage

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
## To do list
- [x] process ECG dataset - AF Classification Challenge 2017
- [x] process ECG dataset - MIT-BIH Arrhythmia Database
- [x] design GAN framework
- [x] hyperparameter model
- [x] mode collapse 
    - [x] apply minibatch discrimination
- [ ] validation

## Output 

<img src="generate_ECG/e4000_7.png" alt="e4000_7.png" width="200"/><img src="generate_ECG/e4000_16.png" alt="e4000_16.png" width="200"/><img src="generate_ECG/e4000_11.png" alt="e4000_11.png" width="200"/><img src="generate_ECG/e4000_19.png" alt="e4000_19.png" width="200"/><img src="generate_ECG/e4000_40.png" alt="e4000_40.png" width="200"/><img src="generate_ECG/e5000_12.png" alt="e5000_12.png" width="200"/><img src="generate_ECG/e5000_26.png" alt="e5000_26.png" width="200"/><img src="generate_ECG/e7000_44.png" alt="e7000_44.png" width="200"/><img src="generate_ECG/e8000_33.png" alt="e8000_33.png" width="200"/><img src="generate_ECG/e10000_51.png" alt="e10000_51.png" width="200"/><img src="generate_ECG/e10000_89.png" alt="e10000_89.png" width="200"/>


