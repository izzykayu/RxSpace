import pandas as pd
import plac
from pathlib import Path
import jsonlines
from datetime import datetime
from sklearn.preprocessing import LabelEncoder



def make_data(d, categories):
    unique_topics = set()
    texts = []
    labels = []
    for i, row in d.iterrows():
        item = row.copy()#dict()
        text = row['text']
        texts.append(text)
        topics = list(row['class'])
    #    print(topics)

        for cat in categories:
            item[cat] = False
            for topic in topics:
                # if topic in unique_topics:
                #     continue
                unique_topics.add(topic)
                if topic == cat:
                    item[cat] = True
        labels.append(item)
        print(unique_topics)
    return labels

plac.annotations(path=('input path', 'i','positional', Path),
                 labels=('labels path', 'l','positional' ,Path),
             #    new_path=('output path', 'o','positional' ,Path),
                 task=('int of task', 't', 'option', int),



          )

def main(path, labelspath):
    train = pd.read_csv(path)
    set_type = path.split('/')[-1].split('.csv')[0]
    new_path = f'{set_type}.jsonl'
    train['text'] = train['unprocessed_text'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
    train['class'] = train['class'].map(lambda x: x.upper())
    labelsdf = pd.get_dummies(train['class'].tolist(), dtype=float)
    today = datetime.now().strftime('%m%y%d-%H%M')
    print(today)
    with open(labelspath, 'r') as f:
        label_data =f.read()
    categories = label_data.split('\n')
    print(f'{len(categories)}')

    # for cat in categories:
    #     textcat.add_label(cat)
    # train = pd.concat([train, labelsdf])
    print(make_data(train.head(), categories=categories))
    ldf = make_data(train, categories=categories)
    print(ldf)
    print(f'writing out df to {new_path}')
    reader = train.to_dict('records')
    c = 0
    with jsonlines.open(new_path, 'w') as writer:
        for obj in reader:
            c += 1
            print(obj)
            break
            # writer.write({
            #     'text': obj['text'],
            #     'cats': [{: ''}],
            #     'meta': {'tweetid': obj.get('tweetid')}})

if __name__ == '__main__':
    plac.call(main)

#example #{"meta": {"id": "000103f0d9cfb60f"}, "text": "D'aww! He matches this background colour I'm seemingly stuck with. Thanks.  (talk) 21:51, January 11, 2016 (UTC)", "cats": {"insult": 0, "obscene": 0, "severe_toxic": 0,  "toxic": 0}}