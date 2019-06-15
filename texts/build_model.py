import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Permute,MaxPooling1D, Flatten, LSTM, Bidirectional, GRU, GlobalAveragePooling1D

from keras.datasets import imdb
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras.optimizers import RMSprop
from keras import backend as K  
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.models import Model
import os, itertools, math 

def construct_original_network(emb, data,trainable=True):
	if data in ['imdbcnn']:
		filters = 250
		kernel_size = 3
		hidden_dims = 250

		net = Dropout(0.2, name = 'dropout_1')(emb)		 

		# we add a Convolution1D, which will learn filters
		# word group filters of size filter_length:
		net = Conv1D(filters,
						 kernel_size,
						 padding='valid',
						 activation='relu',
						 strides=1,
						 name = 'conv1d_1',trainable=trainable)(net)

		# net = Activation('relu', name = 'activation_1')(net)
		# we use max pooling:
		net = GlobalMaxPooling1D(name = 'global_max_pooling1d_1')(net)

		# We add a vanilla hidden layer:
		net = Dense(hidden_dims, name = 'dense_1',trainable=trainable)(net)
		net = Dropout(0.2, name = 'dropout_2')(net)
		net = Activation('relu', name = 'activation_2')(net)

		# We project onto a single unit output layer, and squash it with a sigmoid:
		net = Dense(2, name = 'dense_2',trainable=trainable)(net)
		preds = Activation('softmax', name = 'activation_3')(net)
		return preds


class TextModel():
	def __init__(self, data, train = False):
		self.data = data
		if data in ['imdbcnn']:

			filters = 250 
			hidden_dims = 250
			self.embedding_dims = 50
			self.maxlen = 400
			self.num_classes = 2
			self.num_words = 20002
			self.type = 'word'
			if not train:
				K.set_learning_phase(0)

			X_ph = Input(shape=(self.maxlen,), dtype='int32')
			emb_layer = Embedding(self.num_words, self.embedding_dims,
				input_length=self.maxlen, name = 'embedding_1')
			emb_out = emb_layer(X_ph) 

			if train:
				preds = construct_original_network(emb_out, data)	

			else: 
				emb_ph = Input(shape=(self.maxlen,self.embedding_dims), dtype='float32')   

				preds = construct_original_network(emb_ph, data) 


			if not train:
				model1 = Model(X_ph, emb_out)
				model2 = Model(emb_ph, preds) 
				pred_out = model2(model1(X_ph))  
				pred_model = Model(X_ph, pred_out) 
				pred_model.compile(loss='categorical_crossentropy',
							  optimizer='adam',
							  metrics=['accuracy']) 
				self.pred_model = pred_model 
				grads = []
				for c in range(self.num_classes):
					grads.append(tf.gradients(preds[:,c], emb_ph))

				grads = tf.concat(grads, axis = 0)  
				# [num_classes, batchsize, maxlen, embedding_dims]

				approxs = grads * tf.expand_dims(emb_ph, 0) 
				# [num_classes, batchsize, maxlen, embedding_dims]
				self.sess = K.get_session()  
				self.grads = grads 
				self.approxs = approxs
				self.input_ph = X_ph
				self.emb_out = emb_out
				self.emb_ph = emb_ph
				weights_name = 'original.h5'#[i for i in os.listdir('imdblstm/models/') if i.startswith('original')][0]
				model1.load_weights('{}/models/{}'.format(data, weights_name), 
					by_name=True)
				model2.load_weights('{}/models/{}'.format(data, weights_name), 
					by_name=True)  
				print('Model constructed.')
				# For validating the data. 
				emb_weights = emb_layer.get_weights() 
				emb_weights[0][0] = np.zeros(50)
				emb_layer.set_weights(emb_weights)
			else:
				pred_model = Model(X_ph, preds)
				
				pred_model.compile(loss='categorical_crossentropy',
							  optimizer='adam',
							  metrics=['accuracy']) 
				self.pred_model = pred_model
				from load_data import Data
				dataset = Data(self.data)
				self.train(dataset) 
				print('Training is done.') 


	def train(self, dataset):
		if self.data in ['imdbcnn']:
			epochs = 5
			batch_size = 40
			filepath = '{}/models/original.h5'.format(self.data)
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
				verbose=1, save_best_only=True, mode='max')
			callbacks_list = [checkpoint]
			# embedding_matrix = np.load('data/embedding_matrix.npy')
			self.pred_model.fit(dataset.x_train, dataset.y_train, validation_data=(dataset.x_val, dataset.y_val),callbacks = callbacks_list, epochs=epochs, batch_size=batch_size)


	def predict(self, x, verbose=0):
		if self.data in ['imdbcnn']: 
			if type(x) == list or x.shape[1] < self.maxlen:
				x = np.array(sequence.pad_sequences(x, maxlen=self.maxlen)) 

			return self.pred_model.predict(x, batch_size = 2500, 
				verbose = verbose) 


