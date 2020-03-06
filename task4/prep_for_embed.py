import sys
import re
import csv
from nltk.tokenize import TweetTokenizer

tweet_tokenizer = TweetTokenizer()

def process(sentence):
	sentence = re.sub("@.* ", "_U ", sentence)
	sentence = re.sub("[0-9]", "d", sentence)
	sentence = sentence.lower()
	t_sentence = tweet_tokenizer.tokenize(sentence)
	final = t_sentence.join(" ")
	return final

def load_data(file_to_add, text_col_name, delimiter=','):
	with open(file_to_add, newline='') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			new_text = process(row[text_col_name])
			print(new_text)
			break

#	if file_to_add.endswith('csv'):
#		contents = pd.read_csv(file_to_add, delimiter=delimiter)
#	return contents[text_col_name]

load_data(sys.argv[1], sys.argv[2])








