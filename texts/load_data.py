from keras.datasets import imdb
import numpy as np 
from keras.preprocessing import sequence 
import cPickle as pickle 
import os
import sys
from imdbcnn.imdb_data import create_imdb_dataset


class Data():
	def __init__(self, data, train = False,limited_length = False):
		if data in ['imdbcnn']:
			if 'data' not in os.listdir('./imdbcnn'):
				os.mkdir('./imdbcnn/data')

			data_dir = './imdbcnn/data'
			

			if 'x_val.npy' in os.listdir(data_dir):
				x_val, y_val = np.load('{}/x_val.npy'.format(data_dir)),np.load('{}/y_val.npy'.format(data_dir))
				x_train, y_train = np.load('{}/x_train.npy'.format(data_dir)),np.load('{}/y_train.npy'.format(data_dir))

				with open('{}/id_to_word.pkl'.format(data_dir),'rb') as f:
					self.id_to_word = id_to_word = pickle.load(f)

			else:
				print('Loading data...')
				data, labels, texts, word_index = create_imdb_dataset()
				id_to_word = {value:key for key,value in word_index.items()}

				indices = np.arange(data.shape[0])
				np.random.seed(0)
				np.random.shuffle(indices)

				data = data[indices] 
				texts = texts[indices] 
				labels = labels[indices]
				VALIDATION_SPLIT = 0.1
				nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

				x_train_raw = texts[:-nb_validation_samples]
				x_train = data[:-nb_validation_samples] 
				y_train = labels[:-nb_validation_samples]

				x_val_raw = texts[-nb_validation_samples:]
				x_val = data[-nb_validation_samples:] 
				y_val = labels[-nb_validation_samples:]

				print('Number of positive and negative reviews in traing and validation set')
				print(y_train.sum(axis=0))
				print(y_val.sum(axis=0))
				print(len(x_train), 'train sequences')
				print(len(x_val), 'test sequences')

				print('Pad sequences (samples x time)')


				np.save('{}/x_train.npy'.format(data_dir), x_train)
				np.save('{}/y_train.npy'.format(data_dir), y_train)

				np.save('{}/x_val.npy'.format(data_dir), x_val)
				np.save('{}/y_val.npy'.format(data_dir), y_val)

				np.save('{}/x_train_raw.npy'.format(data_dir), x_train_raw)
				np.save('{}/x_val_raw.npy'.format(data_dir), x_val_raw) 

				with open('imdbcnn/data/id_to_word.pkl','wb') as f:
					pickle.dump(id_to_word, f)	


			self.x_train = x_train
			self.y_train = y_train

			num_data = 300
			self.x_val = x_val if train else x_val[:num_data] 
			self.y_val = y_val if train else y_val[:num_data] 

			self.id_to_word = id_to_word 
			self.val_len = np.load('imdbcnn/data/len_test.npy')
			self.train_len = np.load('imdbcnn/data/len_train.npy')
			self.val_len = np.minimum(self.val_len, 400)
