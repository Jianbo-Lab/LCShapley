# author - Richard Liao 
# Dec 26 2016
import numpy as np
import pandas as pd
try:
	import cPickle as pkl 
except:
	import _pickle as pkl
from collections import defaultdict
import re

from bs4 import BeautifulSoup

import sys
import os
import time
# os.environ['KERAS_BACKEND']='theano'
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.preprocessing import sequence 


from keras import backend as K
from keras.engine.topology import Layer, InputSpec
# from keras import initializations
import json
import tensorflow as tf
MAX_SENT_LENGTH = 100
MAX_SENTS = 15
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2


def clean_str(string):
	"""
	Tokenization/string cleaning for dataset
	Every dataset is lower cased except
	"""
	string = re.sub(r"\\", "", string)    
	string = re.sub(r"\'", "", string)    
	string = re.sub(r"\"", "", string)    
	return string.strip()

def create_imdb_dataset():
	st = time.time()
	print('Loading dataset...')
	data_train = pd.read_csv('imdbcnn/labeledTrainData.tsv', sep='\t')
	data_test = pd.read_csv('imdbcnn/testData.tsv', sep='\t')
	print(data_train.shape)
	print(data_test.shape)


	from nltk import tokenize

	# reviews = []
	labels = []
	texts = []
	lower_texts = []
	# for idx in range(2000):
	for idx in range(data_train.review.shape[0]):
		text = BeautifulSoup(data_train.review[idx])
		text = clean_str(text.get_text().encode('ascii','ignore'))
		texts.append(text) # texts is the raw text
		lower_texts.append(text.lower())
		# sentences = tokenize.sent_tokenize(text)
		# reviews.append(sentences) 
		labels.append(data_train.sentiment[idx])

	for idx in range(data_test.review.shape[0]):
		text = BeautifulSoup(data_test.review[idx])
		text = clean_str(text.get_text().encode('ascii','ignore'))
		texts.append(text) # texts is the raw text
		lower_texts.append(text.lower())
		# sentences = tokenize.sent_tokenize(text)
		# reviews.append(sentences) 
		if data_test.id[idx][-1] in "12345":
			labels.append(0)
		else:
			labels.append(1)

	print("total data", len(texts))

	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lower_texts)
	data = tokenizer.texts_to_sequences(lower_texts)
	data = np.array(data)
	np.save('imdbcnn/data/unreplaced_integers.npy', data) 
	# this is the data without any unknown, a full list of intergers, original sequence

	word_index_for_visualization = tokenizer.word_index
	with open('imdbcnn/data/word_index_for_visualization.pkl','wb') as f:
		pkl.dump(word_index_for_visualization, f)

	data = sequence.pad_sequences(data, maxlen=400)
	data = np.minimum(data, MAX_NB_WORDS + 1) # make the out of 20000 data to be <UNK>



	word_to_id = tokenizer.word_index
	word_index = {key: value for key, value in word_to_id.items() if value <= MAX_NB_WORDS}
	word_index["<PAD>"] = 0
	word_index["<UNK>"] = MAX_NB_WORDS + 1

	print('Total %s unique tokens.' % len(word_index))
	labels = to_categorical(np.asarray(labels))

	texts = np.array(texts)


	print('Creating dataset takes {}s.'.format(time.time()-st))

	print('Storing dataset...')  

	return data, labels, texts, word_index
	





