```python
! pip install gcsfs
! pip install jsonlines
! pip install pandas
! pip install ekphrasis
```

    Requirement already satisfied: gcsfs in /opt/conda/lib/python3.7/site-packages (0.6.0)
    Requirement already satisfied: requests in /opt/conda/lib/python3.7/site-packages (from gcsfs) (2.23.0)
    Requirement already satisfied: google-auth-oauthlib in /opt/conda/lib/python3.7/site-packages (from gcsfs) (0.4.1)
    Requirement already satisfied: fsspec>=0.6.0 in /opt/conda/lib/python3.7/site-packages (from gcsfs) (0.6.2)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.7/site-packages (from gcsfs) (4.4.2)
    Requirement already satisfied: google-auth>=1.2 in /opt/conda/lib/python3.7/site-packages (from gcsfs) (1.11.2)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (1.25.7)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (2019.11.28)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (2.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /opt/conda/lib/python3.7/site-packages (from google-auth-oauthlib->gcsfs) (1.2.0)
    Requirement already satisfied: six>=1.9.0 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (1.14.0)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (3.1.1)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (0.2.7)
    Requirement already satisfied: rsa<4.1,>=3.1.4 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (4.0)
    Requirement already satisfied: setuptools>=40.3.0 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (46.0.0.post20200311)
    Requirement already satisfied: oauthlib>=3.0.0 in /opt/conda/lib/python3.7/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib->gcsfs) (3.0.1)
    Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /opt/conda/lib/python3.7/site-packages (from pyasn1-modules>=0.2.1->google-auth>=1.2->gcsfs) (0.4.8)
    Requirement already satisfied: jsonlines in /opt/conda/lib/python3.7/site-packages (1.2.0)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from jsonlines) (1.14.0)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.7/site-packages (0.25.0)
    Requirement already satisfied: numpy>=1.13.3 in /opt/conda/lib/python3.7/site-packages (from pandas) (1.18.1)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/lib/python3.7/site-packages (from pandas) (2019.3)
    Requirement already satisfied: python-dateutil>=2.6.1 in /opt/conda/lib/python3.7/site-packages (from pandas) (2.8.1)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.7/site-packages (from python-dateutil>=2.6.1->pandas) (1.14.0)
    Requirement already satisfied: ekphrasis in /opt/conda/lib/python3.7/site-packages (0.5.1)
    Requirement already satisfied: ftfy in /opt/conda/lib/python3.7/site-packages (from ekphrasis) (5.7)
    Requirement already satisfied: nltk in /opt/conda/lib/python3.7/site-packages (from ekphrasis) (3.4.4)
    Requirement already satisfied: tqdm in /opt/conda/lib/python3.7/site-packages (from ekphrasis) (4.33.0)
    Requirement already satisfied: ujson in /opt/conda/lib/python3.7/site-packages (from ekphrasis) (2.0.1)
    Requirement already satisfied: matplotlib in /opt/conda/lib/python3.7/site-packages (from ekphrasis) (3.2.0)
    Requirement already satisfied: colorama in /opt/conda/lib/python3.7/site-packages (from ekphrasis) (0.4.3)
    Requirement already satisfied: termcolor in /opt/conda/lib/python3.7/site-packages (from ekphrasis) (1.1.0)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from ekphrasis) (1.18.1)
    Requirement already satisfied: wcwidth in /opt/conda/lib/python3.7/site-packages (from ftfy->ekphrasis) (0.1.8)
    Requirement already satisfied: six in /opt/conda/lib/python3.7/site-packages (from nltk->ekphrasis) (1.14.0)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib->ekphrasis) (2.8.1)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.7/site-packages (from matplotlib->ekphrasis) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib->ekphrasis) (1.1.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/lib/python3.7/site-packages (from matplotlib->ekphrasis) (2.4.6)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->ekphrasis) (46.0.0.post20200311)



```python
from datetime import datetime
import gcsfs

# setting up file system to be ble to read from buckets

fs = gcsfs.GCSFileSystem(project='sm4h-rxspace')

now = datetime.now()
dt = now.strftime('%Y-%m-%d %H:%M')
print(f'start time:\n{dt}')
```

    start time:
    2020-03-25 08:06



```python
# import packages
import jsonlines
import pandas as pd

pd.set_option('display.max_colwidth', 0)


```


```python
def get_distribution(df, col='class'):
    """gives distribution of a column from a pandas data-frame """
    df_out = df[col].value_counts()
    n_train = df.shape[0]
    print(f"loaded {n_train} samples\n")

    df_out = pd.DataFrame(df_out)
    df_out.columns = ['class counts']
    df_out['class %'] = round(100 * df_out['class counts'] / n_train, 2)
    return df_out

```


```python
train_path = "gs://sm4h-rxspace/task4/train.csv"
dev_path = "gs://sm4h-rxspace/task4/validation.csv"
print(f'train path : {train_path}\ndev path : {dev_path}')
```

    train path : gs://sm4h-rxspace/task4/train.csv
    dev path : gs://sm4h-rxspace/task4/validation.csv



```python
df_train_raw = pd.read_csv(train_path)
df_train_raw['class'] = df_train_raw['class'].map(str.strip)

print(f'loaded train from {train_path}')
get_distribution(df_train_raw)
```

    loaded train from gs://sm4h-rxspace/task4/train.csv
    loaded 10537 samples
    





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




```python
df_val_raw = pd.read_csv(dev_path)
df_val_raw['class'] = df_val_raw['class'].map(str.strip)

print(f'loaded dev from {dev_path}..')
get_distribution(df_val_raw)
```

    loaded dev from gs://sm4h-rxspace/task4/validation.csv..
    loaded 2635 samples
    





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




```python
df_train_raw.head()[['tweetid', 'unprocessed_text', 'class']]
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
      <th>unprocessed_text</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1202189293432823810</td>
      <td>_U _U i even see a lot of readmits on those. risperdal consta, abilify maintena, haldol lai, all of them.</td>
      <td>m</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1200504615760023552</td>
      <td>_U valium o clock</td>
      <td>m</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1201776570492489728</td>
      <td>Stop Xanax 😂😂😂😂</td>
      <td>m</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1200528076159029248</td>
      <td>_U tbh it’s the valium i’m on rn prob</td>
      <td>c</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1201420901633400832</td>
      <td>_U i got mine pulled out about 6 years ago and the doctor prescribed me oxycodone but i never had pain. i just got high lol</td>
      <td>a</td>
    </tr>
  </tbody>
</table>
</div>




```python
# loading text preprocessing
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

text_processor = TextPreProcessor(
    # terms that waill be normalized
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

sentences = df_train_raw['unprocessed_text'].tolist()[:11]
sentences
```

    Reading twitter - 1grams ...
    Reading twitter - 2grams ...
    Reading twitter - 1grams ...





    ['_U _U i even see a lot of readmits on those. risperdal consta, abilify maintena, haldol lai, all of them.',
     '_U valium o clock',
     'Stop Xanax 😂😂😂😂',
     '_U tbh it’s the valium i’m on rn prob',
     '_U i got mine pulled out about 6 years ago and the doctor prescribed me oxycodone but i never had pain. i just got high lol',
     "Today is my 1 year vegan anniversary, also over a year since I've used a microwave and 1 year being clean of Vyvanse after struggling being on it for 8 years🤘",
     'Hurricane #Irma upgraded back to a Category-5 storm; maximum sustained winds 160 mph.   Can someone get this chick a xanax or soemthin????',
     "_U I'm on 100mg of Pristiq as well as I'm on Tramadol which boosts serotonin levels, I'm a fucking monster mate",
     'last timethis haopened i spent two months taking morphine and spending50% of mydays bedbound so uh. hoping its just a onw off and not thatagain tbh!',
     '_U _U "innovation isn\'t as likely" is a massive understatement btw. The only popular consumer products that the USSR invented were the Rubik\'s cube, Tetris and Fanta Orange drink. And methadone if you want to include that.',
     "“_U: 99.3% of the world's hydrocodone is used in the united states. 😳 #ascp15 #dea” wow!"]




```python
for s in sentences:
    print(type(s), s)
    print(" ".join(text_processor.pre_process_doc(s)))
```

    <class 'str'> _U _U i even see a lot of readmits on those. risperdal consta, abilify maintena, haldol lai, all of them.
    _u _u i even see a lot of readmits on those . risperdal consta , abilify maintena , haldol lai , all of them .
    <class 'str'> _U valium o clock
    _u valium o clock
    <class 'str'> Stop Xanax 😂😂😂😂
    stop xanax 😂 😂 😂 😂
    <class 'str'> _U tbh it’s the valium i’m on rn prob
    _u tbh it ’ s the valium i ’ m on rn prob
    <class 'str'> _U i got mine pulled out about 6 years ago and the doctor prescribed me oxycodone but i never had pain. i just got high lol
    _u i got mine pulled out about <number> years ago and the doctor prescribed me oxycodone but i never had pain . i just got high lol
    <class 'str'> Today is my 1 year vegan anniversary, also over a year since I've used a microwave and 1 year being clean of Vyvanse after struggling being on it for 8 years🤘
    today is my <number> year vegan anniversary , also over a year since i have used a microwave and <number> year being clean of vyvanse after struggling being on it for <number> years 🤘
    <class 'str'> Hurricane #Irma upgraded back to a Category-5 storm; maximum sustained winds 160 mph.   Can someone get this chick a xanax or soemthin????
    hurricane irma upgraded back to a category - <number> storm ; maximum sustained winds <number> mph . can someone get this chick a xanax or soemthin ? ? ? ?
    <class 'str'> _U I'm on 100mg of Pristiq as well as I'm on Tramadol which boosts serotonin levels, I'm a fucking monster mate
    _u i am on 1 0 0 mg of pristiq as well as i am on tramadol which boosts serotonin levels , i am a fucking monster mate
    <class 'str'> last timethis haopened i spent two months taking morphine and spending50% of mydays bedbound so uh. hoping its just a onw off and not thatagain tbh!
    last timethis haopened i spent two months taking morphine and spending50 % of mydays bedbound so uh . hoping its just a onw off and not thatagain tbh !
    <class 'str'> _U _U "innovation isn't as likely" is a massive understatement btw. The only popular consumer products that the USSR invented were the Rubik's cube, Tetris and Fanta Orange drink. And methadone if you want to include that.
    _u _u " innovation is not as likely " is a massive understatement btw . the only popular consumer products that the ussr invented were the rubik ' s cube , tetris and fanta orange drink . and methadone if you want to include that .
    <class 'str'> “_U: 99.3% of the world's hydrocodone is used in the united states. 😳 #ascp15 #dea” wow!
    “ _u : <percent> of the world ' s hydrocodone is used in the united states . 😳 ascp15 dea ” wow !



```python


def preprocess_tweet_text(s):
    """using ekphrasis preprocessng """
    return " ".join(text_processor.pre_process_doc(s))
```


```python
    
def write_df(df, out_path, text_col='text', label_col='class', metadata=None):
    """
    takes a datafrmae, writes out text col, label col
    """
    
    cnt = 0
    with jsonlines.open(out_path, 'w') as writer:
        for i, row in df.iterrows():
            if metadata is None:
                metadata_res = ''
            metadata_res = row[metadata]
            #tweetid = row['tweetid']
            text = row[text_col]
            text = preprocess_tweet_text(text)
            label = row[label_col]
            # to strip white spaces and etc
            label = label.strip()
            writer.write({
                'text': text,
                'label': label,
                'metadata': metadata,

            })
            
            
            cnt += 1
    print(f"wrote {cnt} lines to {out_path}")
    


    
```


```python
write_df(df_train_raw, out_path='train.jsonl', text_col='unprocessed_text', label_col='class', metadata='tweetid')

```

    wrote 10537 lines to train.jsonl



```python
write_df(df_val_raw, out_path='validation.jsonl', text_col='unprocessed_text', label_col='class', metadata='tweetid')

```

    wrote 2635 lines to validation.jsonl



```python

```


```python

```
