"""
Usage: python prepare-for-spacy-and-pytorch.py <input-dir> <output-dir>

Example: python prepare-for-spacy-pytorch.py data-orig data-spacy-pytorch-jsonl

reading from data-orig/train.csv
writing to data-spacy-pytorch-jsonl/train.jsonl
wrote out 10537 samples from training
reading from data-orig/validation.csv
writing to data-spacy-pytorch-jsonl/validation.jsonl
wrote out 2635 samples from validation

"""
import csv
import jsonlines
import plac
from pathlib import Path
from utilz import listify, preprocess_text

class_dispatch = {
    'a': 'ABUSE',
    'm' : 'MENTION',
    'u' : 'UNRELATED',
    'c': 'CONSUMPTION'
    }

plac.annotations(
    input_dir=("Input directory with csv files", "positional", "i", Path),
    output_dir=("Output directory for preprocessed data now as jsonlines", "positional", "o", Path),
    text_col=("input text column", "option", "t", str),
    label_col=("input label column", "option", "l", str),
    meta_col=("meta information columns", "option", "m", str),
    preprocess=("boolean", "option", "p", bool),
                 )

def main(input_dir, output_dir, text_col='unprocessed_text',
         label_col='class', meta_col="tweetid", preprocess=False):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    #meta_cols =listify(meta_cols)

    train_input_path = Path(input_dir) / 'train.csv'
    print(f'reading from {train_input_path}')

    train_output_path = Path(output_dir) / 'train.jsonl'
    print(f'writing to {train_output_path}')
    cnt = 0
    with jsonlines.open(train_output_path, 'w') as writer:
        with open(train_input_path, 'r') as csvfile:
            tweetreader = csv.DictReader(csvfile, delimiter=',')
            for row in tweetreader:
                cnt += 1
                text = row[text_col]
                if preprocess is True:
                    text = preprocess_text(text)
                tweetid = row[meta_col]
                label = row[label_col]
                new_label = class_dispatch.get(label)
                writer.write({
                    'text': text,
                    'label': new_label,
                    'meta':
                        {
                            'tweetid': tweetid,
                    }
                    })
    print(f'wrote out {cnt} samples from training')

    val_input_path = Path(input_dir) / 'validation.csv'
    print(f'reading from {val_input_path}')

    val_output_path = Path(output_dir) / 'validation.jsonl'
    print(f'writing to {val_output_path}')

    cntv = 0
    with jsonlines.open(val_output_path, 'w') as writer:
        with open(val_input_path, 'r') as csvfile:
            tweetreader = csv.DictReader(csvfile, delimiter=',')
            for row in tweetreader:
                cntv += 1
                text = row[text_col]
                if preprocess is True:
                    text = preprocess_text(text)
                tweetid = row[meta_col]
                label = row[label_col]
                new_label = class_dispatch.get(label)
                writer.write({
                    'text': text,
                    'label': new_label,
                    'meta':
                        {
                            'tweetid': tweetid,
                    }
                    })
    print(f'wrote out {cntv} samples from validation')

if __name__ == '__main__':
    plac.call(main)