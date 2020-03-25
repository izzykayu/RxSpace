import pandas as pd
#import fasttext
import csv
from utilz import preprocess_text


def convert_labels(val):
    val = val.strip(' ')
    return '__label__' + val



def prepare_dataset(path, new_path):
    train = pd.read_csv(path)
#     print(train.head())
    set_type = path.split('/')[-1].split('.csv')[0]
    train['label'] = train['class'].map(convert_labels)
#     print(train.head())
    print(f'label distribution for {set_type} set')
    print(train.label.value_counts())
    print(f'number of samples in {set_type} set')
    print(len(train))
    train['text'] = train['unprocessed_text'].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
    train['text'] = train['text'].map(preprocess_text)
    train_df_fasttext = train[['label', 'text']]
    print(train_df_fasttext.head())
    print(f'writing out df to {new_path}')
    train_df_fasttext.to_csv(f'{new_path}', index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")
    return train_df_fasttext



train_df = pd.read_csv('../data-orig/train.csv')
print(train_df.head())
print(train_df.columns)
print(train_df['class'].map(convert_labels))
train_df_fasttext = prepare_dataset(path='../data-orig/train.csv', new_path='../data-fasttext/task4.train')
val_df_fasttext = prepare_dataset(path='../data-orig/validation.csv', new_path='../data-fasttext/task4.val')
#print(help(fasttext))
# model = fasttext.train_supervised(input="../data-fasttext/task4.train")
# model.save_model("task4baseline.bin")
# tweet_example = val_df_fasttext.text[0]
# print(model.predict(tweet_example))


# why is sourcing data important
# what is the baseline model
# embeddings as input
