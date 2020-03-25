
### Predictions
* This folder `preds` provides the predictions for the competition
* the prediction file is a comma delimited file with the tweetid and clss prediction
* looks as follows (where there is the tweetid,Class)

```text
tweetid,Class
1201409307167862784,m
1200007750383738885,c
1199244035006902272,m
```

```
              precision    recall  f1-score   support

           a       0.55      0.35      0.43       448
           c       0.67      0.69      0.68       730
           m       0.76      0.85      0.80      1353
           u       0.87      0.68      0.76       104

    accuracy                           0.72      2635
   macro avg       0.71      0.64      0.67      2635
weighted avg       0.70      0.72      0.70      2635
```
