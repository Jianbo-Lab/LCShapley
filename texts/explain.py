from __future__ import absolute_import, division, print_function 

import numpy as np
import tensorflow as tf
import os
from keras.utils import to_categorical

import time 
import numpy as np 
import sys
import osimport math
from explain_shapley import explain_shapley, construct_positions_dict_localshapley, construct_positions_connectedshapley
from build_model import TextModel 
from load_data import Data  


def calculate_acc(pred, y):
	return np.mean(np.argmax(pred, axis = 1) == np.argmax(y, axis = 1))



def lcshapley(args):
	dataset, model =  args.dataset, args.model
	st = time.time()
	print('Making explanations...')
	scores = []

	if args.method == 'connectedshapley':
		construct_positions_dict = lambda d,k: construct_positions_connectedshapley(d, k, args.max_order)
	elif args.method in ['localshapley']:
		construct_positions_dict = construct_positions_dict_localshapley

	for i, sample in enumerate(dataset.x_val[:100]):  
		print('explaining the {}th sample...'.format(i)) 

		if model.type == 'word': 
			d = dataset.val_len[i]
			positions_dict, key_to_idx, positions, coefficients, unique_inverse = construct_positions_dict(d, args.num_neighbors)

			sample = sample[-d:]
			inputs = sample * positions 
			print('Explaining inputs size ({},{})'.format(inputs.shape[0],inputs.shape[1]))

		score = explain_shapley(model.predict, sample, d, args.num_neighbors, positions, key_to_idx, inputs, coefficients, unique_inverse)

		scores.append(score)

	np.save('{}/results/scores-{}-{}.npy'.format(args.data, args.method,
			args.num_neighbors), 
			scores)

	print('Time spent is {}'.format(time.time() - st))
	



if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--method', type = str, 
		choices = ['connectedshapley','localshapley','create_predictions','train'], 
		default = 'localshapley')
	parser.add_argument('--data', type = str, 
		choices = ['imdbcnn'], default = 'imdbcnn') 
	parser.add_argument('--num_neighbors', type = int, default = 4) 
	parser.add_argument('--train', action='store_true')
	parser.add_argument('--original', action='store_true')
	parser.add_argument('--max_order', type = int, default = 16)

	args = parser.parse_args()
	dict_a = vars(args)   
	if args.method == 'train':
		model = TextModel(args.data, train = True)


	else:
		print('Loading dataset...') 
		dataset = Data(args.data)

		print('Creating model...')
		model = TextModel(args.data) 

		dict_a.update({'dataset': dataset, 'model': model})

	if args.data not in os.listdir('./'):	
		os.mkdir(args.data)
	if 'results' not in os.listdir('./{}'.format(args.data)):
		os.mkdir('{}/results'.format(args.data))

	if args.method in ['localshapley','connectedshapley']:
		dict_a.update({'regression': False})
		scores = lcshapley(args)






