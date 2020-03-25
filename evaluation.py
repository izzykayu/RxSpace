import plac
import sklearn.metrics as sklm
import pandas as pd
from pathlib import Path



# path to real data -
# File format: csv
# File name: prediction_task4.csv
# Must include columns: tweetid and Class
# *Class is the column that contains the prediction

class_dispatch_fasttext = {
    '__label__a' : 'a',
    '__label__m': 'm',
    '__label__u': 'u',
    '__lavel__c': 'c'
    }

label_dispath = {
    "A": 'potential abuse/misuse',
    "C": 'non-abuse/-misuse consumption',
    "M": 'merely mention the medication',
    "U": 'unrelated'
    }

def clean_orig_labels(val):
    return val.strip(' ')

def convert_(val):
    return val.replace('__label__', '')

def convert_labels_from_fasttext(val):
    val = val.strip(' ')
    return class_dispatch_fasttext.get(val)

plac.annotations(true=('t','true labels', 'option', 't', Path),
                 preds=('p', 'predictions from model', 'option','p', Path),
                 label=('l', 'name of column where that contains the classifcation label for the training set', 'option','l', str))

def main(true='data-orig/validation.csv', preds='preds/fasttextbaseline-predictions.txt', label='class'):
    print(f'reading in data from {true}\nreading in predictions from {preds}')
    true = Path(true)
    df = pd.read_csv(true)
    true_labels = df[label].map(clean_orig_labels)
    true_labels = true_labels.tolist()
    preds = Path(preds)
    y_preds = pd.read_csv(preds, header=None)
    y_preds = y_preds[0].map(convert_)
    y_preds = y_preds.tolist()
    print(sklm.classification_report(y_true=true_labels, y_pred=y_preds))

if __name__ == '__main__':
    plac.call(main)
