"""

python
"""
from utilz import get_top_cat, preprocess_text
import plac
import csv
from pathlib import Path
import spacy


class_dispatch_spacy = {
    'ABUSE': 'a',
    'MENTION': 'm',
    'UNRELATED': 'u',
    'CONSUMPTION': 'c'
    }

class_dispatch_fasttext = {
    '__label__a' : 'a',
    '__label__m': 'm',
    '__label__u': 'u',
    '__lavel__c': 'c'
    }
def preprocess_spacy(label):
    return class_dispatch_spacy.get(label)


plac.annotations(
    input_path=('prediction tweets path', 'i', 'option', Path),
    model_dir=("Optional output directory", "option", "m", Path),
    output_path=("outpush path with predictions", 'option', "o", Path),
    preprocess=("boolean to preprocess", "option", "p", bool),
    model_framework=("model frame work, e.g., spacy,pytorch,keras,fasttext,starspace", "option", "f", str),
                 )

def main(input_path="data-orig/validation.csv", model_dir="spacy-cnn-twitter-glove", output_path="preds/cnn-twitter-validation-predictions.csv", preprocess=True, model_framework="spacy"):
    print(f'loading model from {model_dir}')
    nlp = spacy.load(model_dir)
    cnt = 0
    with open(output_path, 'w') as csvfile_out:
        fieldnames = ['tweetid', 'Class']
        writer = csv.DictWriter(csvfile_out, fieldnames=fieldnames)

        writer.writeheader()
        print(f'loading in tweets from {input_path}')
        with open(input_path, 'r') as csvfile:
            tweetreader = csv.DictReader(csvfile, delimiter=',')
            for row in tweetreader:
                cnt += 1
                tweetid = row['tweetid']
                text = row['unprocessed_text']
                if preprocess is True:
                    text = preprocess_text(text)
                doc = nlp(text)
                # get predicted ckass and score (dont care for score)
                label, _ = get_top_cat(doc)

                new_label = class_dispatch_spacy.get(label)
                writer.writerow({
                    'tweetid': tweetid,
                    'Class': new_label
                    })
                # TODO: add model framework so we can use with all models
    print(f'wrote out {cnt} predictions to {output_path}')

if __name__ == '__main__':
    plac.call(main)
