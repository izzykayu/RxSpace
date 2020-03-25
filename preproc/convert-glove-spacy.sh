
echo "converting glove twitter vectors"

python -m gensim.scripts.glove2word2vec --input "/Users/isabelmetzger/PycharmProjects/glove-twitter/glove.twitter.27B.100d.txt" --output glove.twitter.27B.100d.w2v.txt
gzip glove.twitter.27B.100d.w2v.txt
python -m spacy init-model en twitter-glove --vectors-loc glove.twitter.27B.100d.w2v.txt.gz
