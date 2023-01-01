#!/bin/bash
# vast setup: wget https://v0.bahman.me/train_nlp.tar.gz -c --user=v0 --password='v0v1v2v3'; tar -xzvf train_nlp.tar.gz; chmod +x setup.sh; source setup.sh;
pip3 install -r requirements.txt

mkdir -p objects

dataset_url=http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
wget $dataset_url -c -P datasets
cd datasets
unzip -n $(basename "$dataset_url")
mv training.1600000.processed.noemoticon.csv train.csv
mv testdata.manual.2009.06.14.csv test.csv

cd ..

#glove_url=https://nlp.stanford.edu/data/glove.twitter.27B.zip
glove_url=https://v0.bahman.me/glove-twitter-50.zip
glove_name="glove.twitter.27B.50d.txt"
#wget $glove_url -c -P glove
wget $glove_url -c -P glove --user=v0 --password='v0v1v2v3'
cd glove
unzip -n $(basename "$glove_url")
ln -sf $glove_name glove.txt
