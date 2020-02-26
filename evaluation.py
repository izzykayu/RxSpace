import plac
import sklearn.metrics as sklm
import pandas as pd
from pathlib import Path
def convert_labels(val):
    val = val.strip(' ')
    return '__label__' + val

plac.annotations(true=('t','true labels', 'positional', Path),
                 preds=('p', 'predictions from model', 'positional', Path),
                 label=('l', 'name of label', 'optional', str))
def main(true, preds, label='class'):
    df = pd.read_csv(true)
    true_labels = df[label].map(convert_labels).values
    y_preds = pd.read_csv(preds, header=None)
    y_preds = y_preds[0].values
    print(sklm.classification_report(y_true=true_labels, y_pred=y_preds))

if __name__ == '__main__':
    plac.call(main)
