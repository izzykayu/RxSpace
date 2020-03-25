"""
Usage: eval-official.py <gt_pth> <pred_pth>
takes in the ground truth (e.g., tweetid, class,


"""
import sklearn.metrics as sklm
import pandas as pd
import plac
from pathlib import Path

def clean_val(val):
    return val.strip(' ')

plac.annotations(gt_pth=("ground truth path to csv file - must have columns tweetid and class", "positional", "t", Path),
                 pred_pth=("prediction classes csv file - mustve have columns tweetid and Class", "positional", "p", Path),
                 )

def main(gt_pth, pred_pth):
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

    truths = truth_df['class'].map(clean_val)
    truths = truths.tolist()
    preds = pred_df['Class']
    preds = preds.tolist()
    print(sklm.classification_report(y_true=truths, y_pred=preds))

if __name__ == '__main__':
    plac.call(main)
