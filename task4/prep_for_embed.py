import sys
import re
import csv
import nltk.data
from nltk.tokenize import TweetTokenizer

sentence_detector = nltk.data.load('tokenizers/punkt/english.pickle')
tweet_processor = TweetTokenizer()


def write_to_emb(sentence):
	emb = open('./data/emb.txt', 'a')
	emb.write(sentence + '\n')
	emb.close()

def process(line):
	line = line.strip()
	sentences = sentence_detector.tokenize(line)
	for sentence in sentences:
		sentence = re.sub(' @.* ', ' _U ', sentence)
		sentence = re.sub('[0-9]', 'd', sentence) 
		sentence = sentence.lower()
		t_sentence = tweet_processor.tokenize(sentence)
		final = " ".join(t_sentence)
		write_to_emb(final)
	return final

def load_data(file_to_add, text_col_name, delimiter=','):
	with open(file_to_add, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			new_text = process(row[text_col_name])

#	if file_to_add.endswith('csv'):
#		contents = pd.read_csv(file_to_add, delimiter=delimiter)
#	return contents[text_col_name]

load_data(sys.argv[1], sys.argv[2])








