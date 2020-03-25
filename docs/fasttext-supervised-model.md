### text preprocessing foor fastText
* fastText for multiclass input data looks like a flat file delimited with ```__label__ + label_class + <\space> + preprocessed_text```


```python
! pip install fasttext
! pip install pandas
! pip install gcsfs

```

    Requirement already satisfied: fasttext in /opt/conda/lib/python3.7/site-packages (0.9.1)
    Requirement already satisfied: numpy in /opt/conda/lib/python3.7/site-packages (from fasttext) (1.18.1)
    Requirement already satisfied: pybind11>=2.2 in /opt/conda/lib/python3.7/site-packages (from fasttext) (2.4.3)
    Requirement already satisfied: setuptools>=0.7.0 in /opt/conda/lib/python3.7/site-packages (from fasttext) (45.2.0.post20200209)
    Requirement already satisfied: pandas in /opt/conda/lib/python3.7/site-packages (1.0.1)
    Requirement already satisfied: pytz>=2017.2 in /opt/conda/lib/python3.7/site-packages (from pandas) (2019.3)
    Requirement already satisfied: numpy>=1.13.3 in /opt/conda/lib/python3.7/site-packages (from pandas) (1.18.1)
    Requirement already satisfied: python-dateutil>=2.6.1 in /opt/conda/lib/python3.7/site-packages (from pandas) (2.8.1)
    Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.7/site-packages (from python-dateutil>=2.6.1->pandas) (1.14.0)
    Requirement already satisfied: gcsfs in /opt/conda/lib/python3.7/site-packages (0.6.0)
    Requirement already satisfied: google-auth-oauthlib in /opt/conda/lib/python3.7/site-packages (from gcsfs) (0.4.1)
    Requirement already satisfied: requests in /opt/conda/lib/python3.7/site-packages (from gcsfs) (2.23.0)
    Requirement already satisfied: google-auth>=1.2 in /opt/conda/lib/python3.7/site-packages (from gcsfs) (1.11.2)
    Requirement already satisfied: fsspec>=0.6.0 in /opt/conda/lib/python3.7/site-packages (from gcsfs) (0.6.2)
    Requirement already satisfied: decorator in /opt/conda/lib/python3.7/site-packages (from gcsfs) (4.4.1)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /opt/conda/lib/python3.7/site-packages (from google-auth-oauthlib->gcsfs) (1.2.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (1.25.7)
    Requirement already satisfied: idna<3,>=2.5 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (2.9)
    Requirement already satisfied: chardet<4,>=3.0.2 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.7/site-packages (from requests->gcsfs) (2019.11.28)
    Requirement already satisfied: setuptools>=40.3.0 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (45.2.0.post20200209)
    Requirement already satisfied: cachetools<5.0,>=2.0.0 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (3.1.1)
    Requirement already satisfied: rsa<4.1,>=3.1.4 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (4.0)
    Requirement already satisfied: six>=1.9.0 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (1.14.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/conda/lib/python3.7/site-packages (from google-auth>=1.2->gcsfs) (0.2.7)
    Requirement already satisfied: oauthlib>=3.0.0 in /opt/conda/lib/python3.7/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib->gcsfs) (3.0.1)
    Requirement already satisfied: pyasn1>=0.1.3 in /opt/conda/lib/python3.7/site-packages (from rsa<4.1,>=3.1.4->google-auth>=1.2->gcsfs) (0.4.8)



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

    starting at 2020-03-25 09:20



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
    "CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
    "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
    "@SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
]

for s in sentences:
    print(type(s), s)
    print(" ".join(text_processor.pre_process_doc(s)))
```

    Reading twitter - 1grams ...
    Reading twitter - 2grams ...
    Reading twitter - 1grams ...
    <class 'str'> CANT WAIT for the new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))
    cant wait for the new season of twin peaks ＼(^o^)／ ! ! ! david lynch tv series <happy>
    <class 'str'> I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/
    i saw the new john doe movie and it suuuuucks ! ! ! waisted <money> . . . bad movies <annoyed>
    <class 'str'> @SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/.
    <user> : can not wait for the <date> sentiment talks ! yaaaaaay ! ! ! <laugh> <url>



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
import csv

def main(inpath, outpath, text_col='unprocessed_text', label_col='class'):
    
    df = pd.read_csv(inpath)
    n = len(df)
    print(f"read in {n} samples from {inpath}")
    
    df['label'] = df[label_col].map(create_fasttext_label)
    df['text'] = df[text_col].replace('\n', ' ', regex=True).replace('\t', ' ', regex=True)
    df['text'] = df['text'].map(str)
    df['text'] = df['text'].map(preprocess_fasttext)
    fasttext_df = df[['label', 'text']]
    fasttext_df.to_csv(f"{outpath}", index=False, sep=' ',
                       header=False, quoting=csv.QUOTE_NONE,
                      quotechar="", escapechar=" ")
    print(f"wrote out fasttext prepared text to {outpath}")
    
    
    
    
```


```python

train_pth = "gs://sm4h-rxspace/task4/train.csv"
dev_pth = "gs://sm4h-rxspace/task4/validation.csv"


```


```python
main(inpath=train_pth, outpath="fastText-0.9.1/data/tweets-fasttext.train")

```

    read in 10537 samples from gs://sm4h-rxspace/task4/train.csv
    wrote out fasttext prepared text to fastText-0.9.1/data/tweets-fasttext.train



```python
main(inpath=dev_pth, outpath="fastText-0.9.1/data/tweets-fasttext.dev")
```

    read in 2635 samples from gs://sm4h-rxspace/task4/validation.csv
    wrote out fasttext prepared text to fastText-0.9.1/data/tweets-fasttext.dev



```python

model = fasttext.train_supervised(input='fastText-0.9.1/data/tweets-fasttext.train',
                                  lr=0.5, epoch=25,
                                  wordNgrams=2,
                                  bucket=200000,
                                  dim=100,
                                  loss='ova')
```


```python
model.save_model('fasttext_model_tweets.bin')
```


```python

verbose_map = {
    'a': 'ABUSE',
    'm': 'MENTION',
    'u': 'UNRELATED',
    'c': 'CONSUMPTION'
              }

def predict_twitter(inpath, outpath, text_col='unprocessed_text', label_col='class', n_samples=10):
    
    df = pd.read_csv(inpath)
    if n_samples is None:
        n_samples = len(df)
        
    df = df.sample(n_samples)

    print(f"read in {n_samples} samples from {inpath}")
    
    df['label'] = df[label_col].map(create_fasttext_label)
    df['text'] = df[text_col].replace('\n', ' ', regex=True).replace('\t', ' ', regex=True)
    df['text'] = df['text'].map(str)
    df['text'] = df['text'].map(preprocess_fasttext)
    preds_list = []
    for i, row in df.iterrows():
        tweetid = row['tweetid']
        text = row['text']
        print(text)
        pred_lb, score = model.predict(text)
        pred = pred_lb[0].replace('__label__', '')
        print(f'pred class: {verbose_map.get(pred)}\npred score {round(score[0], 4)}')
        true_label = row[label_col]
        print(f'true class: {verbose_map.get(true_label)}')
        preds_list.append({'tweetid': tweetid,
                          'Class': pred})
        
    fasttext_df = pd.DataFrame(preds_list)
    fasttext_df.to_csv(f"{outpath}", index=False, quoting=csv.QUOTE_NONE,
                      quotechar="", escapechar=" ")
    print(f"wrote out fasttext prepared text to {outpath}")
        

    
    
```


```python
predict_twitter(inpath=train_pth, outpath='preds-validation-fasttext-twitter-model-samples.csv', n_samples=20)
```

    read in 20 samples from gs://sm4h-rxspace/task4/train.csv
    <time> and i am already on tramadol 😞 🔫
    pred class: CONSUMPTION
    pred score 0.9996
    true class: CONSUMPTION
    day <number> of being sober from addy and xanax and feelin good 😇 gonna keep this up idc how difficult it gets
    pred class: ABUSE
    pred score 0.9962
    true class: ABUSE
    if it ’ s not too late can someone give xan to a couple of xanax so it can calm down a bit before tomorrow ?
    pred class: MENTION
    pred score 0.9922
    true class: MENTION
    _u oh c ' mon , that ' s knudepunkt but on valium and with business cards . . .
    pred class: MENTION
    pred score 1.0
    true class: MENTION
    life of a college kid : " i am about to take an adderall and drink a beer with it . " - _u
    pred class: MENTION
    pred score 0.9935
    true class: MENTION
    i changed for each one of my adderall high where i feel like mario after he ' s found a star .
    pred class: ABUSE
    pred score 1.0
    true class: ABUSE
    so done with this adderall 😒
    pred class: CONSUMPTION
    pred score 1.0
    true class: CONSUMPTION
    _u _u no we should make big pharma fund substance abuse treatment centers for heavily marketing oxycontin to drs and hospitals and covering up the addictive properties of the painkillers they pedal that have exacerbated the drug problem we have . next .
    pred class: MENTION
    pred score 1.0
    true class: MENTION
    _u _u it ' s worse than that . ibogaine can cure heroin addiction much more efficiently than methadone . it ' s legal in europe & asia . but , the patent on it expired . the drug company could not make money off it , so they never applied for fda approval .
    pred class: MENTION
    pred score 1.0
    true class: MENTION
    on the block i am magical see me at ya college campus baggy full of adderall
    pred class: MENTION
    pred score 0.9909
    true class: MENTION
    _u maybe like an adderall like substance that helps them concentrate but is banned ? 🤷
    pred class: MENTION
    pred score 1.0
    true class: MENTION
    a british citizen has been jailed for <number> years in egypt for taking tramadol in to there country as its illegal there ( shock as shit ) but it ’ s there rules and this is why you need to no about where your traveling to especially with medication .
    pred class: MENTION
    pred score 0.9958
    true class: MENTION
    so . i might be opening a big ol ' can of worms here , but i ' d like info about tramadol , esp as relates to eds and mcas patients . thinking about switching over from as - needed to a scheduled regimen .
    pred class: MENTION
    pred score 0.9927
    true class: MENTION
    man that hydrocodone be having me out for hellas
    pred class: CONSUMPTION
    pred score 1.0
    true class: CONSUMPTION
    _u rt _u : adderall had me cleaning my mirrors when all i wanted to do was take a selfie
    pred class: MENTION
    pred score 0.9993
    true class: MENTION
    _u on codeine , still got naproxen and some diazepam left . she said i shouldnt have been prescribed as much as i did . i ain ' t ( <money>
    pred class: CONSUMPTION
    pred score 0.9959
    true class: CONSUMPTION
    the valium is probably helping too .
    pred class: CONSUMPTION
    pred score 1.0
    true class: CONSUMPTION
    saw a doctor to get my anxiety treated , was applauded for my article . surreal . also i have xanax now thank goodness
    pred class: CONSUMPTION
    pred score 1.0
    true class: CONSUMPTION
    best line of the trip yet . me : i can ’ t find my ativan ! ! ! mrs : how about some red licorice ? me : how about some prayers to st . anthony ! ! ! ativan found , all will soon be well in brian - land !
    pred class: CONSUMPTION
    pred score 0.9943
    true class: CONSUMPTION
    does anybody have any good recipes for adderall
    pred class: ABUSE
    pred score 0.9997
    true class: ABUSE
    wrote out fasttext prepared text to preds-validation-fasttext-twitter-model-samples.csv



```python

```
