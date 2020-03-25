## SM4H - Team **RxSpace**!


## Table of Contents
* [Competition Details](#competition-details)
* [Team Members](#team)
* [Our Approach](#our-approach)
* [Text Corpora](#text-corpora)
* [Requirements](#requirements)
* [Repo Setup](#repo-setup)
* [Word Embeddings](#embeddings)
* [Snorkel](#snorkel)
* [Model Training](#model-training)
* [Evaluation](#evaluation)
* [References](#references)
* [Tags](#tags)


## Competition Details
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


## Team
### Team members
* Isabel Metzger - im1247@nyu.edu <br>
* Allison Black - aab3711@gmail.com <br>
* Rajat Chandra - rajatsc4@gmail.com <br>
* Rishi Bhargava - rishi.bhargava42@gmail.com <br>
* Emir Haskovic - emir.y.haskovic@gmail.com <br> 
* Mark Rutledge - mark.t.rutledge@gmail.com <br>
* Natasha Zaliznyak - nzaliznyak@gmail.com
* Whitley Yi - wmcadenhead@gmail.com <br>

## Our Approach
* 

## Text Corpora
### Supervised Learning
* Original train/validation split:
   * We use the train.csv, validation.csv as provided from our competition
    train size = 10537 samples

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class counts</th>
      <th>class %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m</th>
      <td>5488</td>
      <td>52.08</td>
    </tr>
    <tr>
      <th>c</th>
      <td>2940</td>
      <td>27.90</td>
    </tr>
    <tr>
      <th>a</th>
      <td>1685</td>
      <td>15.99</td>
    </tr>
    <tr>
      <th>u</th>
      <td>424</td>
      <td>4.02</td>
    </tr>
  </tbody>
</table>
</div>
    validation/dev: 2635 samples
<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class counts</th>
      <th>class %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>m</th>
      <td>1353</td>
      <td>51.35</td>
    </tr>
    <tr>
      <th>c</th>
      <td>730</td>
      <td>27.70</td>
    </tr>
    <tr>
      <th>a</th>
      <td>448</td>
      <td>17.00</td>
    </tr>
    <tr>
      <th>u</th>
      <td>104</td>
      <td>3.95</td>
    </tr>
  </tbody>
</table>
</div>

* Multiple Splits:
   * For our ensemble method of multiple text classification models, we train models on different splits (70:30) of shuffled and stratified by class combined train + val <br>
   
### Unsupervised Learning
We created word embeddings using health social media posts from twitter and other public datasets. We used [ekphrasis]( https://github.com/cbaziotis/ekphrasis) and nltk tweet tokenizer for tokenization and sentencizing. Preprocessing can be found in the preprocessing notebook.
 

| Sources  | Sentences/Tweets | Tokens |
| :------  | --------: | -----: |
| Twitter (SM4H) |  |  | 
| Drug Reviews| | |
|  Wikipedia |  |  | 
| | | |

## Requirements
* Important packages/frameworks utilized include [spacy](https://github.com/explosion/spaCy), [fastText](https://github.com/facebookresearch/fastText), [ekphrasis](https://github.com/cbaziotis/ekphrasis), [allennlp](https://github.com/allenai/allennlp), [PyTorch](https://github.com/pytorch/pytorch), [snorkel](https://github.com/snorkel-team/snorkel/)
* To use the allennlp configs (nlp_cofigs/text_classification.json) with pre-trained scibert embeddings, which were downloaded as below 
```bash
wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/pytorch_models/scibert_scivocab_uncased.tar
tar -xvf scibert_scivocab_uncased.tar
```
* Exact requirements can be found in the requirements.txt file
* For specific processed done in jupyter notebooks, please find the packages listed in the beginning cells of each notebook

## Embeddings


## Repo Layout
```
* rx_twitterspace
* nlp_configs
* preds
* data-orig
* docs
* saved-models
```

## Model Training
* baseline fastText supervised classifier
* allennlp frameworks




## Evaluation

### Text classification
* Run `python eval-official.py` to see the evaluation on predictions made from our fasttext baseline model which preprocessed text using ekphrasis

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
Out of the box with fasttext.train_supervised(tweets.train)
```bash

              precision    recall  f1-score   support

           a       0.59      0.27      0.37       448
           c       0.65      0.68      0.67       730
           m       0.74      0.88      0.80      1353
           u       0.87      0.58      0.69       104

    accuracy                           0.71      2635
   macro avg       0.71      0.60      0.63      2635
weighted avg       0.70      0.71      0.69      2635



```
#converting glove twitter vectors

```bash
python -m gensim.scripts.glove2word2vec --input "/Users/isabelmetzger/PycharmProjects/glove-twitter/glove.twitter.27B.100d.txt" --output glove.twitter.27B.100d.w2v.txt
gzip glove.twitter.27B.100d.w2v.txt
python -m spacy init-model en twitter-glove --vectors-loc glove.twitter.27B.100d.w2v.txt.gz

```


## Tags
* data augmentation, weak supervision, noisy labeling, word embeddings, text classification, multi-label, multi-class, scalability