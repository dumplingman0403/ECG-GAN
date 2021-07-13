#!/bin/sh

# sudo brew install unzip

wget -O training2017.zip https://physionet.org/files/challenge-2017/1.0.0/training2017.zip?download
unzip training2017.zip
mv training2017 AF_dataset
rm -f training2017.zip


wget -O mit-bih-arrhythmia-database-1.0.0.zip https://physionet.org/static/published-projects/mitdb/mit-bih-arrhythmia-database-1.0.0.zip?download
unzip mit-bih-arrhythmia-database-1.0.0.zip
mv mit-bih-arrhythmia-database-1.0.0 AA_dataset
rm -f mit-bih-arrhythmia-database-1.0.0.zip