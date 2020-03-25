### model predictions
* fastText for multiclass input data looks like a flat file delimited with ```__label__ + label_class + <\space> + preprocessed_text```
* here in this notebook we load an already trained model to get all model predictions for all classes


```python
! pip install fasttext
! pip install pandas
! pip install gcsfs
! pip install jsonlines
```

    Requirement already satisfied: fasttext in /opt/conda/lib/python3.7/site-packages (0.9.1)
    Requirement already satisfied: setuptools>=0.7.0 in /opt/conda/lib/python3.7/site-packages (from fasttext) (45.2.0.post20200209)
    Requirement already satisfied: pybind11>=2.2 in /opt/conda/lib/python3.7/site-packages (from fasttext) (2.4.3)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from fasttext) (1.18.1)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.7/site-packages (1.0.1)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/lib/python3.7/site-packages (from pandas) (2019.3)
    Requirement already satisfied: python-dateutil>=2.6.1 in /opt/conda/lib/python3.7/site-packages (from pandas) (2.8.1)
    Requirement already satisfied: numpy>=1.13.3 in /opt/conda/lib/python3.7/site-packages (from pandas) (1.18.1)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.7/site-packages (from python-dateutil>=2.6.1->pandas) (1.14.0)
    Requirement already satisfied: gcsfs in /opt/conda/lib/python3.7/site-packages (0.6.0)
    Requirement already satisfied: google-auth-oauthlib in /opt/conda/lib/python3.7/site-packages (from gcsfs) (0.4.1)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.7/site-packages (from gcsfs) (4.4.1)
    Requirement already satisfied: requests in /opt/conda/lib/python3.7/site-packages (from gcsfs) (2.23.0)
    Requirement already satisfied: google-auth>=1.2 in /opt/conda/lib/python3.7/site-packages (from gcsfs) (1.11.2)
    Requirement already satisfied: fsspec>=0.6.0 in /opt/conda/lib/python3.7/site-packages (from gcsfs) (0.6.2)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /opt/conda/lib/python3.7/site-packages (from google-auth-oauthlib->gcsfs) (1.2.0)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (1.25.7)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (2.9)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (2019.11.28)
    Requirement already satisfied: rsa<4.1,>=3.1.4 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (4.0)
    Requirement already satisfied: six>=1.9.0 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (1.14.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (0.2.7)
    Requirement already satisfied: setuptools>=40.3.0 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (45.2.0.post20200209)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (3.1.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /opt/conda/lib/python3.7/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib->gcsfs) (3.0.1)
    Requirement already satisfied: pyasn1>=0.1.3 in /opt/conda/lib/python3.7/site-packages (from rsa<4.1,>=3.1.4->google-auth>=1.2->gcsfs) (0.4.8)
    Requirement already satisfied: jsonlines in /opt/conda/lib/python3.7/site-packages (1.2.0)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from jsonlines) (1.14.0)



```python
# importing packgs and creating filespace
import gcsfs
import fasttext
import pandas as pd
import string

fs = gcsfs.GCSFileSystem(project='sm4h-rxspace')
```


```python
from datetime import datetime

dt = datetime.now().strftime('%Y-%m-%d %H:%M')
print(f"starting at {dt}")
```

    starting at 2020-03-25 16:02



```python
# creating text_preprocessing with ekphrasis
import re
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={},
#     annotate={"hashtag", "allcaps", "elongated", "repeated",
#         'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 
    
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 
    
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words
    
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

sentences = [
'Can somebody tell me what morphine is for?', '1.2 mg .02 of my x 6 i feeeeeeeeeeel goooooooood like im on suboxone or methadone', 'Oh hello crippling anxiety, let’s drive home from Pittsburgh shall we?! #ativan', 'Pop a adderall we gone fuck all night', '#fakeexercisefacts a xanax a day keeps the trainer away']
for s in sentences:
    print(type(s), s)
    print(" ".join(text_processor.pre_process_doc(s)))
```

    Reading twitter - 1grams ...
    Reading twitter - 2grams ...
    Reading twitter - 1grams ...
    <class 'str'> Can somebody tell me what morphine is for?
    can somebody tell me what morphine is for ?
    <class 'str'> 1.2 mg .02 of my x 6 i feeeeeeeeeeel goooooooood like im on suboxone or methadone
    <number> mg . <number> of my x <number> i feeeeeeeeeeel goooooooood like im on suboxone or methadone
    <class 'str'> Oh hello crippling anxiety, let’s drive home from Pittsburgh shall we?! #ativan
    oh hello crippling anxiety , let ’ s drive home from pittsburgh shall we ? ! ativan
    <class 'str'> Pop a adderall we gone fuck all night
    pop a adderall we gone fuck all night
    <class 'str'> #fakeexercisefacts a xanax a day keeps the trainer away
    fake exercise facts a xanax a day keeps the trainer away



```python


def preprocess_fasttext(s, lower=True):
    tokens = text_processor.pre_process_doc(s)
    if lower:
        return ' '.join([t.lower() for t in tokens])

    return ' '.join(tokens)
```


```python
def create_fasttext_label(val):
    val = str(val)
    val = val.strip()
    return '__label__' + val

def preprocess_fasttext(s, lower=True):
    tokens = text_processor.pre_process_doc(s)
    if lower:
        return ' '.join([t.lower() for t in tokens])

    return ' '.join(tokens)
```


```python

train_pth = "gs://sm4h-rxspace/task4/train.csv"
dev_pth = "gs://sm4h-rxspace/task4/validation.csv"
model_pth = "fasttext_model_tweets.bin"
```


```python
model = fasttext.load_model(model_pth)

```

    



```python
verbose_map = {
    'a': 'ABUSE',
    'm': 'MENTION',
    'u': 'UNRELATED',
    'c': 'CONSUMPTION'
              }

def prediction_matrix(dev_pth, outpath,text_col='unprocessed_text', label_col='class', n_samples = None):
    
    df = pd.read_csv(dev_pth)
    if n_samples is None:
        n_samples = len(df)
    df = df.head(n_samples)
#print(list(df.unprocessed_text[:5]))
    print(df.head())

    print(f"read in {n_samples} samples from {dev_pth}")
    
    df['label'] = df[label_col].map(create_fasttext_label)
    df['text'] = df[text_col].replace('\n', ' ', regex=True).replace('\t', ' ', regex=True)
    df['text'] = df['text'].map(str)
    df['text'] = df['text'].map(preprocess_fasttext)

    preds_list = []
    for i, row in df.iterrows():
        tweetid = row['tweetid']
        text = row['text']
        true_lab = row[label_col]
     #print(text)
        # to predict all scores
        pred_lbs, scores = model.predict(text, k=4)
        y_pred, score_pred = model.predict(text)
        y_pred = y_pred[0]
        pred_lbs = [lb.replace('__label__', '') for lb in pred_lbs]
        predictions_dict = dict(zip(pred_lbs, scores))
        predictions_dict['text'] = text
        predictions_dict['tweetid'] = tweetid
        predictions_dict['y_true'] = true_lab
        predictions_dict['y_pred'] = y_pred.replace('__label__', '')
    
        preds_list.append(predictions_dict)
    df_pred = pd.DataFrame(preds_list)
    print(df_pred.columns)
    df_pred[['tweetid', 'text', 'y_true', 'y_pred', 'a', 'c', 'm', 'u']].to_csv(f'{outpath}', index=False)
    return df_pred


```


```python

df_pred = prediction_matrix(dev_pth=dev_pth, outpath='prediction-matrix-fasttext.csv')

    
```

       Unnamed: 0 class              tweetid  \
    0        9293     m  1201409307167862784   
    1        4651     m  1200007750383738885   
    2        2275     c  1199244035006902272   
    3        3740     m  1199782125609902084   
    4        3759     m  1199783941764517889   
    
                                        unprocessed_text  
    0         Can somebody tell me what morphine is for?  
    1  1.2 mg .02 of my x 6 i feeeeeeeeeeel goooooooo...  
    2  Oh hello crippling anxiety, let’s drive home f...  
    3              Pop a adderall we gone fuck all night  
    4  #fakeexercisefacts a xanax a day keeps the tra...  
    read in 2635 samples from gs://sm4h-rxspace/task4/validation.csv
    Index(['m', 'u', 'a', 'c', 'text', 'tweetid', 'y_true', 'y_pred'], dtype='object')



```python
df_pred = pd.read_csv('prediction-matrix-fasttext.csv')

pd.set_option('display.max_colwidth', 0)
abuse_subset_fasttext_model = df_pred[df_pred.y_true=='a']
abuse_subset_fasttext_model
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweetid</th>
      <th>text</th>
      <th>y_true</th>
      <th>y_pred</th>
      <th>a</th>
      <th>c</th>
      <th>m</th>
      <th>u</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1198463912330645504</td>
      <td>_u i lied about being claustrophobic for no reason and they gave me a clonazepam shot so it was awesome</td>
      <td>a</td>
      <td>c</td>
      <td>0.007826</td>
      <td>1.000010</td>
      <td>0.000010</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1202352927446380544</td>
      <td>anyone selling some morphine i need like &lt;number&gt; dozes</td>
      <td>a</td>
      <td>a</td>
      <td>0.980886</td>
      <td>0.000010</td>
      <td>0.787941</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1200680073751408640</td>
      <td>looking for some adderall . shoot me the link .</td>
      <td>a</td>
      <td>a</td>
      <td>0.812877</td>
      <td>0.000010</td>
      <td>0.056662</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1199987793482653696</td>
      <td>" do not tell mikes dad i did drugs . he will lock me up . morphine yea hell really lock me up " mikes dad who was a police officer</td>
      <td>a</td>
      <td>m</td>
      <td>0.000010</td>
      <td>0.067557</td>
      <td>0.685959</td>
      <td>0.001180</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1199134018807181312</td>
      <td>got my adderall fix and i feel great ! 😁 😄</td>
      <td>a</td>
      <td>c</td>
      <td>0.012442</td>
      <td>1.000010</td>
      <td>0.000010</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2589</th>
      <td>1201141807071072256</td>
      <td>seroquel munchies are absolutely real i always thought it was a joke</td>
      <td>a</td>
      <td>m</td>
      <td>0.000010</td>
      <td>0.011697</td>
      <td>0.808077</td>
      <td>0.000921</td>
    </tr>
    <tr>
      <th>2599</th>
      <td>1200499901160968194</td>
      <td>how much xanax is too much ? i have gobbled down &lt;number&gt; percocets in a day . got a high tolerance for shit like this . i have heard someone say it ' s possible to smoke too much weed , but i never found a limit myself . good at drugs</td>
      <td>a</td>
      <td>a</td>
      <td>0.981463</td>
      <td>0.000010</td>
      <td>0.021625</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>2607</th>
      <td>1199822537892519936</td>
      <td>he took methadone by mistake and nearly died from it ? how bruh ? !</td>
      <td>a</td>
      <td>m</td>
      <td>0.000010</td>
      <td>0.000010</td>
      <td>1.000010</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>2619</th>
      <td>1199047791454171138</td>
      <td>_u i ’ ve done oxy , hydrocodone , and codeine and can confirm this is accurate</td>
      <td>a</td>
      <td>c</td>
      <td>0.000010</td>
      <td>0.912446</td>
      <td>0.020342</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>2625</th>
      <td>1200499351031894016</td>
      <td>i am on entirely way too much xanax to function during this workout</td>
      <td>a</td>
      <td>c</td>
      <td>0.006498</td>
      <td>0.341593</td>
      <td>0.024433</td>
      <td>0.000010</td>
    </tr>
  </tbody>
</table>
<p>447 rows × 8 columns</p>
</div>




```python

wrong_preds_abuse = abuse_subset_fasttext_model[abuse_subset_fasttext_model.y_pred != 'a']
wrong_preds_abuse.y_pred.value_counts()
```




    m    171
    c    116
    u    3  
    Name: y_pred, dtype: int64




```python
wrong_preds_abuse
        
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweetid</th>
      <th>text</th>
      <th>y_true</th>
      <th>y_pred</th>
      <th>a</th>
      <th>c</th>
      <th>m</th>
      <th>u</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>1198463912330645504</td>
      <td>_u i lied about being claustrophobic for no reason and they gave me a clonazepam shot so it was awesome</td>
      <td>a</td>
      <td>c</td>
      <td>0.007826</td>
      <td>1.000010</td>
      <td>0.000010</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1199987793482653696</td>
      <td>" do not tell mikes dad i did drugs . he will lock me up . morphine yea hell really lock me up " mikes dad who was a police officer</td>
      <td>a</td>
      <td>m</td>
      <td>0.000010</td>
      <td>0.067557</td>
      <td>0.685959</td>
      <td>0.001180</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1199134018807181312</td>
      <td>got my adderall fix and i feel great ! 😁 😄</td>
      <td>a</td>
      <td>c</td>
      <td>0.012442</td>
      <td>1.000010</td>
      <td>0.000010</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1202443453885489152</td>
      <td>i was shunned into never doing coke so like what is morphine then i ’ m down</td>
      <td>a</td>
      <td>c</td>
      <td>0.065615</td>
      <td>0.348655</td>
      <td>0.000010</td>
      <td>0.048868</td>
    </tr>
    <tr>
      <th>62</th>
      <td>1201315490381082624</td>
      <td>_u mikael found himself dumbfounded and at loss for words . is that alright ? will she freak out ? i mean , she offered it herself . scratch that , i will just snort my line of morphine , per usual . mikael dragged his hand to hang up , but his fingers —</td>
      <td>a</td>
      <td>m</td>
      <td>0.000010</td>
      <td>0.000010</td>
      <td>0.999579</td>
      <td>0.006498</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2551</th>
      <td>1200742999656976386</td>
      <td>can ’ t cope with danny off his head on morphine laid in bed watching 9 0 s music videos and</td>
      <td>a</td>
      <td>m</td>
      <td>0.000010</td>
      <td>0.000698</td>
      <td>0.996527</td>
      <td>0.001998</td>
    </tr>
    <tr>
      <th>2589</th>
      <td>1201141807071072256</td>
      <td>seroquel munchies are absolutely real i always thought it was a joke</td>
      <td>a</td>
      <td>m</td>
      <td>0.000010</td>
      <td>0.011697</td>
      <td>0.808077</td>
      <td>0.000921</td>
    </tr>
    <tr>
      <th>2607</th>
      <td>1199822537892519936</td>
      <td>he took methadone by mistake and nearly died from it ? how bruh ? !</td>
      <td>a</td>
      <td>m</td>
      <td>0.000010</td>
      <td>0.000010</td>
      <td>1.000010</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>2619</th>
      <td>1199047791454171138</td>
      <td>_u i ’ ve done oxy , hydrocodone , and codeine and can confirm this is accurate</td>
      <td>a</td>
      <td>c</td>
      <td>0.000010</td>
      <td>0.912446</td>
      <td>0.020342</td>
      <td>0.000010</td>
    </tr>
    <tr>
      <th>2625</th>
      <td>1200499351031894016</td>
      <td>i am on entirely way too much xanax to function during this workout</td>
      <td>a</td>
      <td>c</td>
      <td>0.006498</td>
      <td>0.341593</td>
      <td>0.024433</td>
      <td>0.000010</td>
    </tr>
  </tbody>
</table>
<p>290 rows × 8 columns</p>
</div>




```python

```
