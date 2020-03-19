## SM4H - Team **RxSpace**!

### DETAILS
*This repository contains code for tackling Task 4 of the SMM2020 

The Social Media Mining for Health Applications (#SMM4H) Shared Task involves natural language processing (NLP) challenges of using social media data for health research, including informal, colloquial expressions and misspellings of clinical concepts, noise, data sparsity, ambiguity, and multilingual posts. For each of the five tasks below, participating teams will be provided with a set of annotated tweets for developing systems, followed by a three-day window during which they will run their systems on unlabeled test data and upload the predictions of their systems to CodaLab. Informlsation about registration, data access, paper submissions, and presentations can be found below.
<br>

*Task 4: Automatic characterization of chatter related to prescription medication abuse in tweets* <br>

This new, multi-class classification task involves distinguishing, among tweets that mention at least one prescription opioid, benzodiazepine, atypical anti-psychotic, central nervous system stimulant or GABA analogue, tweets that report potential abuse/misuse (annotated as “A”) from those that report non-abuse/-misuse consumption (annotated as “C”), merely mention the medication (annotated as “M”), or are unrelated (annotated as “U”)3. <br>

#### Timeline
* Training data available: January 15, 2020 (may be sooner for some tasks) <br>
* Test data available: April 2, 2020 <br>
System predictions for test data due: April 5, 2020 (23:59 CodaLab server time) <br>
* System description paper submission deadline: May 5, 2020 <br>
* Notification of acceptance of system description papers: June 10, 2020 <br>
* Camera-ready papers due: June 30, 2020 <br>
* Workshop: September 13, 2020 <br>
* All deadlines, except for system predictions (see above), are 23:59 UTC (“anywhere on Earth”). <br>

## Repo Layout
```


```

## META
### Team members
* Isabel Metzger - im1247@nyu.edu <br>
* Allison Black - aab3711@gmail.com <br>
* Rajat Chandra - rajatsc4@gmail.com <br>
* Rishi Bhargava - rishi.bhargava42@gmail.com <br>
* Emir Haskovic - emir.y.haskovic@gmail.com <br> 
* Mark Rutledge - mark.t.rutledge@gmail.com <br>
* Natasha Zaliznyak - nzaliznyak@gmail.com
* Whitley Yi - wmcadenhead@gmail.com <br>



tags: data augmentation, snorkel labeling functions, elmo, cnns, scalability

### Evaluating predictions 
* fasttext model `python evaluation.py`
```

              precision    recall  f1-score   support

           a       0.59      0.27      0.37       448
           c       0.65      0.68      0.67       730
           m       0.74      0.88      0.80      1353
           u       0.87      0.58      0.69       104

    accuracy                           0.71      2635
   macro avg       0.71      0.60      0.63      2635
weighted avg       0.70      0.71      0.69      2635


              precision    recall  f1-score   support

           a       0.19      0.08      0.11       448
           c       0.28      0.23      0.25       730
           m       0.52      0.69      0.59      1353
           u       0.08      0.02      0.03       104

    accuracy                           0.43      2635
   macro avg       0.27      0.26      0.25      2635
weighted avg       0.38      0.43      0.39      2635


```
#converting glove twitter vectors

```bash
python -m gensim.scripts.glove2word2vec --input "/Users/isabelmetzger/PycharmProjects/glove-twitter/glove.twitter.27B.100d.txt" --output glove.twitter.27B.100d.w2v.txt
gzip glove.twitter.27B.100d.w2v.txt
python -m spacy init-model en twitter-glove --vectors-loc glove.twitter.27B.100d.w2v.txt.gz

```