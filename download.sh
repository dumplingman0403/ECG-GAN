#!/bin/sh
wget -O training2017.zip https://physionet.org/files/challenge-2017/1.0.0/training2017.zip?download
unzip training2017.zip
mv training2017 AF_dataset
rm -f training2017.zip