from itertools import chain, combinations
# from keras.utils.np_utils import to_categorical 
from keras.utils import to_categorical 
import numpy as np
import itertools
import scipy.special
from time import time

def powerset(iterable, order = None):
	"""
	Find the powerset of a set.
	Input:
	iterable: a set represented by list. 
	Output:
	a dictionary that classifies subsets by cardinality. Keys are the cardinality and values are lists of subsets. 
	"""
	if order is None:
		order = len(iterable)

	return {r:list(combinations(iterable, r)) for r in range(order+1)}
	
def convert_index_to_categorical(indices, d): 
	"""
	Convert indices to zero-one vectors.

	Inputs:
	indices: numpy array of shape (n,j). Each row represents nonzero indices
	of a single vector.
	d: The dimension of each output vector.

	Outputs:
	(n,d) array. Each row is a zero-one vector with j ones indexed by 
	the jth row of indices.
	""" 

	a,b = indices.shape 
	if b == 0:
		return np.zeros((a, d))
	else:
		indices = np.reshape(indices, [-1])
		cats = to_categorical(indices, num_classes=d) 
		cats = np.reshape(cats,(a,b,d)) 
		return np.sum(cats, axis = -2) 


def construct_positions_dict_localshapley(d, k): 
	while k >= d:
		k -= 2

	max_order = k+1 # number of elements, including i.
	# recipical of coefficients.
	# coefficients = {j:(1.0/scipy.special.binom(k, j-1))/float(k+1) for j in range(1, k+2)}
	coefficients = {j:(1.0/scipy.special.binom(k, j-1))/float(k+1) for j in range(1, max_order+1)}

	# Construct collection of subsets for each feature.
	subsets = {}
	
	for i in range(d):  
		subset = list(np.array(list(range(i-k/2,i))+list(range(i+1,i+k/2+1))) % d)
		subsets[i] = powerset(subset, max_order - 1) 

	positions_dict ={(i,j,fill):[] for i in range(d) for j in range(1,max_order+1) for fill in [0,1]}

	for i in subsets:
		# pad positions outside S by 1. 
		one_pads = 1 - np.sum(to_categorical(np.array(range(i-k/2,i+k/2+1)) % d, num_classes = d),axis = 0)
		for j in subsets[i]: # j is the size of pos_excluded. j+1 is the size of pos_included.
			for arr in subsets[i][j]:
				# positions_dict[(i,j,0)] = 
				pos_excluded = np.sum(to_categorical(arr, num_classes = d), axis = 0)
				# pad positions outside S by 1.
				
				pos_excluded += one_pads 
				pos_included = pos_excluded + to_categorical(i, num_classes = d)
				positions_dict[(i,j+1,1)].append(pos_included)
				positions_dict[(i,j+1,0)].append(pos_excluded) 
	# values is a list of lists of zero-one vectors.
	keys, values = positions_dict.keys(), positions_dict.values()  
		
	# concatenate a list of lists to a list. 
	# positions = np.array(list(itertools.chain.from_iterable(values)))
	values = [np.array(value) for value in values] 
	positions = np.concatenate(values, axis = 0)

	key_to_idx = {}
	count = 0
	for i, key in enumerate(keys):
		key_to_idx[key] = list(range(count, count + len(values[i])))
		count += len(values[i])

	# reduce the number of func evaluation by removing duplicate.
	print('checking uniqueness...')
	positions, unique_inverse = np.unique(positions, axis = 0, return_inverse = True) 

	# positions = 1 - positions 

	return positions_dict, key_to_idx, positions, coefficients, unique_inverse

def construct_positions_connectedshapley(d, k, max_order = None):

	# Construct collection of subsets for each feature.
	subsets = {}

	while k >= d:
		k -= 2

	if max_order == None:
		max_order = k+1

	# coefficients.
	coefficients = {j:2.0 / (j*(j+1)*(j+2)) for j in range(1,max_order + 1)}
	# max_order = 4 # k+1
	for i in range(d):
		subsets[i] = [np.array(range(i-s,i+t+1)) % d for s in range(k/2+1) for t in range(k/2+1)]
		subsets[i] = [subset for subset in subsets[i] if len(subset) <= max_order]
	# Construct dictionary of indices for points where 
	# the predict is to be evaluated. 
	positions_dict ={(i,l,fill):[] for i in range(d) for l in range(1,max_order+1) for fill in [0,1]}
	for i in subsets:
		# pad positions outside S by 1. 
		one_pads = 1 - np.sum(to_categorical(np.array(range(i-k/2,i+k/2+1)) % d, num_classes = d), axis = 0)

		for arr in subsets[i]:  
			# For feature i, the list of subsets of size j, with/without feature i. 
			l = len(arr) 
			pos_included = np.sum(to_categorical(arr, num_classes = d), axis = 0)
			# pad positions outside S by 1.
			pos_included += one_pads  
			pos_excluded = pos_included - to_categorical(i, num_classes = d) 
			positions_dict[(i,l,1)].append(pos_included)
			positions_dict[(i,l,0)].append(pos_excluded)

	# values is a list of lists of zero-one vectors.
	keys, values = positions_dict.keys(), positions_dict.values() 
	
	# concatenate a list of lists to a list. 
	values = [np.array(value).reshape(-1,d) for value in values]  
	positions = np.concatenate(values, axis = 0)

	key_to_idx = {}
	count = 0
	for i, key in enumerate(keys):
		key_to_idx[key] = list(range(count, count + len(values[i])))
		count += len(values[i])

	# reduce the number of func evaluation by removing duplicate.
	print('checking uniqueness...')
	positions, unique_inverse = np.unique(positions, axis = 0, return_inverse = True) 
	# positions = 1 - positions 

	return positions_dict, key_to_idx, positions, coefficients, unique_inverse


def explain_shapley(predict, x, d, k, positions, key_to_idx, inputs, coefficients, unique_inverse):
	"""
	Compute the importance score of each feature of x for the predict.

	Inputs:
	predict: a function that takes in inputs of shape (n,d), and 
	outputs the distribution of response variable, of shape (n,c),
	where n is the number of samples, d is the input dimension, and
	c is the number of classes.

	x: input vector (d,)

	k: number of neighbors taken into account for each feature. 

	Outputs:
	phis: importance scores of shape (d,)
	"""
	while k >= d:
		k -= 2
	st1 = time()
	# Evaluate predict at inputs.  
	f_vals = predict(inputs) # [n, c]

	probs = predict(np.array([x])) # [1, c]
	
	st2 = time()
	log_probs = np.log(f_vals+np.finfo(float).resolution)

	discrete_probs = np.eye(len(probs[0]))[np.argmax(probs, axis = -1)]
	vals = np.sum(discrete_probs * log_probs, axis = 1) 

	# key_to_idx[key]: list of indices in original position. 
	# unique_inverse[idx]: maps idx in original position to idx in the current position. 
	key_to_val = {key: np.array([vals[unique_inverse[idx]] for idx in key_to_idx[key]]) for key in key_to_idx}

	# Compute importance scores. 
	phis = np.zeros(d)
	for i in range(d):
		phis[i] = np.sum([coefficients[j]*np.sum(key_to_val[(i,j,1)]-key_to_val[(i,j,0)]) for j in coefficients]) 

	st3 = time()
	print('func evaluation: {}s, post processing {}s'.format(st2-st1,st3-st2))
	return phis

