import pandas as pd 
import numpy as np 
import cPickle as pickle
import os 
import csv

def get_selected_words(x_single, score, id_to_word, k): 
	selected_words = {} # {location: word_id}
	maxlen = len(x_single)
	selected = np.argsort(score)[-k:] 
	selected_k_hot = np.zeros(maxlen)
	selected_k_hot[selected] = 1.0

	selected_words = {i: x_single[i] for i in selected}
	list_words = {}
	for i in sorted(selected):
		if i in selected_words:
			word = id_to_word[selected_words[i]]
			if i-1 in list_words:
				list_words[i] = ' '.join([list_words.pop(i-1),word])
			else:
				list_words[i] = word

	x_selected = (x_single * selected_k_hot).astype(int)
	return x_selected, list_words.values()

def create_texts_and_dataset_from_score(method, x, scores, pred, y, training_data, k, part_i =-1):
	
	with open('data/id_to_word.pkl','rb') as f:
		id_to_word = pickle.load(f)
	new_data = []
	new_texts = []
	for i, x_single in enumerate(x):
		x_selected, list_words = get_selected_words(x_single, 
			scores[i], id_to_word, k)

		new_data.append(x_selected)
		new_texts.append('; '.join(list_words))

	true_labels = np.argmax(y, axis = 1)
	original_model_predictions = np.argmax(pred, axis = 1)

	if 'results' not in os.listdir('./'):
		os.mkdir('results')

	ending = '-{}'.format(part_i) if part_i != -1 else ''
	np.save('data/x_{}-{}{}.npy'.format('train' if training_data else 'val', method, ending), np.array(new_data))
	with open('results/{}-{}-words{}.csv'.format(method, 'train' if training_data else 'val',ending), 'wb') as f:
		writer = csv.writer(f)
		for i in range(len(scores)): 
			# print(new_texts[i])
			writer.writerow(["\"" + new_texts[i].encode('utf-8') + "\""])
	# with open('results/{}-{}-words{}.txt'.format(method,'train' if training_data else 'val',ending),'wb') as f:
		# for i in range(len(scores)):
		# 	f.write('{}&{}&{}\n'.format(true_labels[i], original_model_predictions[i], new_texts[i].encode('utf-8'))) 

def compute_lor(prob, cls):
	return np.log(prob[:,cls] + 1e-6) - np.log(1 - prob[:,cls] + 1e-6)	
	

def compute_acc_and_lor_single(score, model, x, original_prob, pys, sign = True):
	if not sign:
		score = abs(score)

	selected_points = []
	masked_points = [] 

	d = len(score)  
	# if model.data == 'agccnn':
	# 	nums = np.arange(1, 1 + d)
	# elif model.data in ['imdbcnn','yahoolstm','yahoomultilstm']:
	# 	nums = np.concatenate((np.arange(1,10),np.arange(10, 400, 10),np.arange(391,401)))

	nums = np.arange(1, 1 + d)

	# nums = np.arange(1,11) 
	for k in nums: 
		selected = np.argsort(score)[-k:] # indices of largest k score.  
		selected_k_hot = np.zeros(d)
		selected_k_hot[selected] = 1.0 

		selected_points.append(selected_k_hot) # selecting largest k. 
		masked_points.append(1 - selected_k_hot) # masking largest k.  

	selected_points, masked_points = np.array(selected_points), np.array(masked_points)

	if model.type == 'word':
		selected_points, masked_points = x[-d:] * selected_points, x[-d:] * masked_points
	elif model.type == 'char':
		words = np.array(model.tokenize(x)) 
		spaces = np.array([' '] * d)
		selected_points = np.where(selected_points != 0, words, spaces)
		masked_points = np.where(masked_points != 0, words, spaces)
		selected_points = [model.detokenize(single_input, return_str = True) for single_input in selected_points]
		masked_points = [model.detokenize(single_input, return_str = True) for single_input in masked_points]

	# print(selected_points.shape)
	probs_selected = model.predict(selected_points) 
	
	probs_masked = model.predict(masked_points) 

	acc_selected = np.argmax(original_prob)==np.argmax(probs_selected, axis = -1)
	acc_masked = np.argmax(original_prob)==np.argmax(probs_masked, axis = -1) 
	# print(acc_selected)
	# only evaluate the original class. 
	cls = np.argmax(original_prob)
	# Decrease of the original class log ratio by selecting a subset of classes.
	lor_change_selected = compute_lor(probs_selected, cls) - compute_lor(original_prob[None], cls)  
	# Increase of the opposite class log ratio by masking a subset of classes. 
	lor_change_masked = compute_lor(probs_masked, cls) - compute_lor(original_prob[None], cls)

	vmi_selected = compute_vmi(original_prob, probs_selected, pys)
	vmi_masked = compute_vmi(original_prob, probs_masked, pys)

	return acc_selected, acc_masked, lor_change_selected, lor_change_masked, vmi_selected, vmi_masked

import cPickle as pkl 
def compute_acc_and_lor(scores, model, xs, probs, method, args, sign = True):
	metrics = ['acc_selected', 'acc_masked', 'lor_change_selected', 'lor_change_masked','vmi_selected', 'vmi_masked'] 
	outputs = {metric: [] for metric in metrics}

	pys = np.mean(probs, axis = 0) 

	for i, score in enumerate(scores):
		print('Validating the {}th sample...'.format(i))
		if i < len(xs):
			x = xs[i]; prob = probs[i] 
			results = compute_acc_and_lor_single(score, model, x, prob, pys, sign = True)  

			for j, metric in enumerate(metrics):
				outputs[metric].append(results[j]) 

	avg_acc = np.mean([out[np.minimum(9,len(out) - 1)] for out in outputs['acc_selected']])
	print('The selected accuracy with ten features is {}'.format(avg_acc))

	with open('{}/results/metrics-{}.pkl'.format(args.data, method),'wb') as f:
		pkl.dump(outputs, f) 




















