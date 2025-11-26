import pandas as pd
import os
import string
from collections import Counter
import math

TRAIN_DATA_PATH = os.path.join("data", "fake_reviews.csv")

def load_data(data_path: str) -> tuple():
	# count for number of times words appear in each category
	cg_tokens = Counter() # fake (computer generated)
	or_tokens = Counter() # real
	# variables tracking how many of each category appear in the data
	cg_count = 0
	or_count = 0
	# initialize dataframe to store data
	df = pd.read_csv(data_path)

	for index, row in df.iterrows(): # iterate through the rows of dataframe
		row_label = row['label'] # get the CG, OR label
		text = row['text_'] # get the text as a string
		text_lower = text.lower() # convert to all lowercase
		#remove punctuation
		translator = str.maketrans('', '', string.punctuation)
		clean_text = text_lower.translate(translator)
		# split cleaned text
		words = clean_text.split()
		# fill token count dictionary for probability calcs
		if row_label == "CG":
			cg_tokens.update(words)
			cg_count += 1
		else:
			or_tokens.update(words)
			or_count += 1

	return cg_tokens, or_tokens, cg_count, or_count

def calc_probs(cg_tokens, or_tokens, cg_count, or_count):
	vocab = set(cg_tokens.keys()).union(set(or_tokens.keys())) # set of unique vocab across both categories

	cg_word_count =  sum(cg_tokens.values()) # total number of words
	or_word_count = sum(or_tokens.values())

	prob_cg = math.log(cg_count / (cg_count + or_count))# probability of just being real or fake
	prob_or = math.log(or_count / (cg_count + or_count))

	cg_probs = dict() # probabilities that each word is in each category
	or_probs = dict()

	for word in vocab:
		if word not in cg_tokens:
			cg_tokens[word] = 0

		cg_probs[word] = math.log((cg_tokens[word] + 1) / (cg_word_count + len(vocab)))# add constant of 1 for laplace smoothing

		if word not in or_tokens:
			or_tokens[word] = 0

		or_probs[word] = math.log((or_tokens[word] + 1) / (or_word_count + len(vocab))) # add constant of 1 for laplace smoothing

	return prob_cg, prob_or, cg_probs, or_probs

def predict(text, prob_cg, prob_or, cg_probs, or_probs):
	cg_prob = prob_cg
	or_prob = prob_or

	text_lower = text.lower()  # convert to all lowercase
	# remove punctuation
	translator = str.maketrans('', '', string.punctuation)
	clean_text = text_lower.translate(translator)
	# split cleaned text
	words = clean_text.split()

	for word in words:
		if word not in cg_probs.keys():
			pass
		else:
			cg_prob += cg_probs[word] # adding due to log probabilities (log(a) + log(b) = log(ba)

		if word not in or_probs.keys():
			pass
		else:
			or_prob += or_probs[word]

	if cg_prob > or_prob:
		return "CG"
	else:
		return "OR"






