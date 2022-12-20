pip3 install -r requirements.txt

dataset_url=http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
wget $dataset_url -c -P datasets
cd datasets
unzip -n $(basename "$dataset_url")
mv training.1600000.processed.noemoticon.csv train.csv
mv testdata.manual.2009.06.14.csv test.csv

cd ..

glove_url=https://nlp.stanford.edu/data/glove.twitter.27B.zip
glove_name="glove.twitter.27B.200d.txt"
wget $glove_url -c -P glove
cd glove
unzip -n $(basename "$glove_url")
ln -sf $glove_name glove.txt