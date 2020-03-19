wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
python -m spacy init-model en /tmp/vectors_wiki_fasttext --vectors-loc wiki.en.vec