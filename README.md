## SM4H - Team **RxSpace**!


## Table of Contents
* [Competition Details](#competition-details)
* [Team Members](#team)
* [Our Approach](#our-approach)
* [Text Corpora](#text-corpora)
* [Requirements](#requirements)
* [Repo Layout](#repo-layout)
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
* *Our approach can be broken up into 3 main sections: preprocessing, model architecture, and voting*
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


## Repo Layout
```
* notebooks - jupyter notebooks including notebooks that contain important steps including embedding preprocessing, preprocessing for our allennlp models, snorkel labeling fxns and evaluation/exploratory analysis, and our baseline fasttext model (preprocessing, training, and saving): process-emb.ipynb, preprocessing-jsonl.ipynb, snorkel.ipynb, fasttext-supervised-model.ipynb
* rx_twitterspace - allennlp library with our dataset loaders, predictors, and models
* nlp_configs - allennlp model experiment configurations
* preds - directory with predictions
* data-orig - directory with original raw data as provided from the SMM4H official task
* docs - more documentation (md and html files)
* saved-models - directory where saved models are
* preproc - bash scripts with import setup and pre-processing bash scripts such as converting fasttext embeddings for spacy and for compiling fasttext library
```
## Embeddings

## Snorkel
### Labeling Fxns
* We used the snorkel framework for two major tasks: labeling Fxns and data augmentation
* labeling function creation [Notebook](https://github.com/izzykayu/RxSpace/blob/master/notebooks/snorkel.ipynb)
# TODO: add link
* data augmentation [notebook]()


## Model Training
* baseline fastText supervised classifier
   * [Notebook](https://github.com/izzykayu/RxSpace/blob/master/notebooks/fasttext-supervised-model.ipynb)
* allennlp + PyTorch frameworks
 * model1
 * [configuration](https://github.com/izzykayu/RxSpace/blob/master/nlp_configs/text_classification.json)
 * To run the model training with this configuration:
 ```bash
 allennlp train nlp_configs/text_classification.json --serialization-dir saved-models/<your-model-dir> --include-package rx_twitterspace
 ```
 * Experiments ran so far include using exactly what is in nlp_configs/text_classification.json, where I have the data preprocessed in [noteboooks](https://github.com/izzykayu/RxSpace/blob/master/notebooks/preprocessing-jsonl.ipynb) in a directory called `data-classification-jsonl` and using the validation metric of best average F1 across all classes
 ```bash
 allennlp train nlp_configs/text_classification.json --serialization-dir saved-models/model1 --include-package rx_twitterspace
 ```
 * end std logging of training:
 ```bash
 2020-03-25 06:51:35,274 - INFO - allennlp.models.archival - archiving weights and vocabulary to saved-models/model1/model.tar.gz
2020-03-25 06:51:55,764 - INFO - allennlp.common.util - Metrics:
```
```json {
  "best_epoch": 8,
  "peak_cpu_memory_MB": 1759.670272,
  "training_duration": "4:16:26.044395",
  "training_start_epoch": 0,
  "training_epochs": 17,
  "epoch": 17,
  "training_m_P": 0.9747639894485474,
  "training_m_R": 0.9783163070678711,
  "training_m_F1": 0.9765369296073914,
  "training_c_P": 0.9627350568771362,
  "training_c_R": 0.9578231573104858,
  "training_c_F1": 0.9602728486061096,
  "training_a_P": 0.923259973526001,
  "training_a_R": 0.9210682511329651,
  "training_a_F1": 0.9221628308296204,
  "training_u_P": 0.9810874462127686,
  "training_u_R": 0.9787735939025879,
  "training_u_F1": 0.9799291491508484,
  "training_average_F1": 0.9597254395484924,
  "training_accuracy": 0.9634620859827275,
  "training_loss": 0.10317539691212446,
  "training_cpu_memory_MB": 1759.670272,
  "validation_m_P": 0.8063355088233948,
  "validation_m_R": 0.8277900815010071,
  "validation_m_F1": 0.8169219493865967,
  "validation_c_P": 0.7048114538192749,
  "validation_c_R": 0.7424657344818115,
  "validation_c_F1": 0.7231488227844238,
  "validation_a_P": 0.5392670035362244,
  "validation_a_R": 0.4598214328289032,
  "validation_a_F1": 0.4963855445384979,
  "validation_u_P": 0.8315789699554443,
  "validation_u_R": 0.7596153616905212,
  "validation_u_F1": 0.7939698100090027,
  "validation_average_F1": 0.7076065316796303,
  "validation_accuracy": 0.7388994307400379,
  "validation_loss": 1.2185947988406722,
  "best_validation_m_P": 0.8156182169914246,
  "best_validation_m_R": 0.8337028622627258,
  "best_validation_m_F1": 0.8245614171028137,
  "best_validation_c_P": 0.6991150379180908,
  "best_validation_c_R": 0.7575342655181885,
  "best_validation_c_F1": 0.7271531820297241,
  "best_validation_a_P": 0.5498652458190918,
  "best_validation_a_R": 0.4553571343421936,
  "best_validation_a_F1": 0.49816855788230896,
  "best_validation_u_P": 0.8666666746139526,
  "best_validation_u_R": 0.75,
  "best_validation_u_F1": 0.8041236996650696,
  "best_validation_average_F1": 0.7135017141699791,
  "best_validation_accuracy": 0.7449715370018976,
  "best_validation_loss": 0.8092704885695354,
  "test_m_P": 0.8156182169914246,
  "test_m_R": 0.8337028622627258,
  "test_m_F1": 0.8245614171028137,
  "test_c_P": 0.6991150379180908,
  "test_c_R": 0.7575342655181885,
  "test_c_F1": 0.7271531820297241,
  "test_a_P": 0.5498652458190918,
  "test_a_R": 0.4553571343421936,
  "test_a_F1": 0.49816855788230896,
  "test_u_P": 0.8666666746139526,
  "test_u_R": 0.75,
  "test_u_F1": 0.8041236996650696,
  "test_average_F1": 0.7135017141699791,
  "test_accuracy": 0.7449715370018976,
  "test_loss": 0.8023609224572239
}
 ```


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