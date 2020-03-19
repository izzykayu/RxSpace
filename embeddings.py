import plac
from pathlib import Path

plac.annotations(embedding_path=("path to embeddings", "positional", "e", Path),
                 embed_type=("embedding type", "positional", "t", str),
                 )

def main(embedding_path, embed_type):
    if embed_type == 'glove':
        pass