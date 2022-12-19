pip3 install -r requirements.txt

mkdir datasets

glove_url=https://nlp.stanford.edu/data/glove.twitter.27B.zip
glove_name="glove.twitter.27B.200d.txt"
wget $glove_url -c -P glove
cd glove
unzip -n $(basename "$glove_url")
ln -s $glove_name glove.txt