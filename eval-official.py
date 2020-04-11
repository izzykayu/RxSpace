"""
Usage: eval-official.py <gt_pth> <pred_pth>
takes in the ground truth (e.g., tweetid, class)
usage: eval-official.py [qs-adderral_lang-en_until-2020-02-02.csv] [gt_pth] [pred_pth] [truth_label] [pred_label]

positional arguments:
  gt_pth       [data-orig/validation.csv]
  pred_pth     [preds/preds-validation-fasttext-twitter-model.csv]
  truth_label  [class]
  pred_label   [Class]

optional arguments:
  qs-adderral_lang-en_until-2020-02-02.csv, --help   show this help message and exit

"""
import sklearn.metrics as sklm
import pandas as pd
import plac
from pathlib import Path


plac.annotations(gt_pth=("ground truth path to csv file - must have columns tweetid and class", "option", "gt", Path),
                 pred_pth=("prediction classes csv file - mustve have columns tweetid and Class","option", "p", Path),
                 truth_label=("column name for ground truth - e.g., class",  "option", "l",  str),
                 pred_label=("column name for prediction.csv predicted class column name, e.g., Class", "option", "pl", str),
                 )

def main(gt_pth='data-orig/validation.csv',
         pred_pth='preds/preds-validation-fasttext-twitter-model.csv', truth_label='class', pred_label='Class'):
    truth_df = pd.read_csv(gt_pth)
    tweet_ids_truths = truth_df['tweetid']
    tweet_ids_truths = tweet_ids_truths.tolist()

    pred_df = pd.read_csv(pred_pth)
    tweet_ids_preds = pred_df['tweetid']
    tweet_ids_preds = tweet_ids_preds.tolist()

    print('Checking to ensure that tweet ids for ground truths and for predictions are equal')
    yes = tweet_ids_preds==tweet_ids_truths
    if yes is True:
        print(f'{yes}')
    if yes is False:
        print(f'are you sure you put in the correct paths?')

    truths = truth_df[truth_label].map(str.strip)
    truths = truths.tolist()
    preds = pred_df[pred_label].map(str.strip)
    preds = preds.tolist()
    print(sklm.classification_report(y_true=truths, y_pred=preds))

if __name__ == '__main__':
    plac.call(main)
