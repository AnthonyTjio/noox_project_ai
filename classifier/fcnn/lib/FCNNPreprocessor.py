import sys
from datetime import datetime
import re

import tables
import random
from random import randint
import numpy as np
import json
import fasttext

from .CSVReader import CSVReader
from .ListManipulator import ListManipulator
from .StringManipulator import StringManipulator
from .FCNNConfig import FCNNConfig

class FCNNPreprocessor:
	config = FCNNConfig()

	max_word_count = config.model.max_word_count

	word_vector_size = config.model.frame_size

	labels = config.model.labels
	labels_length = len(labels)

	@classmethod
	def get_content_data(cls, np_data, content_rows=[1,2]):
		data =  np_data[:, content_rows]
		return data

	@classmethod
	def normalize_content_data(cls, np_data, stemming=False, remove_stopword=False):
		if stemming:
			for i, dum in enumerate(np_data):
				print("Normalizing #{}".format(i))
				for j, dum in enumerate(dum):
					np_data[i,j] = str(np_data[i,j])
					np_data[i,j] = StringManipulator.normalize_text(np_data[i,j], remove_stopword=remove_stopword)
		else:
			for i, dum in enumerate(np_data):
				print("Normalizing #{}".format(i))
				for j, dum in enumerate(dum):
					np_data[i,j] = str(np_data[i,j])
					np_data[i,j] = np_data[i,j].lower()
					if remove_stopword:
						np_data[i,j] = StringManipulator.remove_stopwords(np_data[i,j])
	
		return np_data

	@classmethod
	def convert_dataset(cls, np_data, content_rows=[1,2], label_row=3, reverse=True, convert_to_vector=True):
		# Merge content_rows and remove all non-content and non-label rows
		content_file = tables.open_file('fcnn_input.h5', mode="w")
		label_file = tables.open_file('fcnn_label.h5', mode="w")
		x_inputs= content_file.create_vlarray(content_file.root, 
													'fcnn_input', 
													tables.Int8Atom(shape=(cls.word_vector_size, cls.max_word_count)), 
													"fcnn inputs", 
													filters=tables.Filters(1))
		y_labels = label_file.create_vlarray(label_file.root,
												'fcnn_label',
												tables.Int8Atom(shape=(cls.labels_length)),
												'fcnn lables',
												filters=tables.Filters(1))	
		model = fasttext.load_model(cls.config.word_model.model_dir)

		for i, rows in enumerate(np_data):
			print("Merging #{}".format(i))
			content = ''
			label = None

			# Combine content rows
			for j, column in enumerate(rows): 
				if(j in content_rows): 
					content += column # Merge Selected Contents
				elif(j == label_row):
					label = column # Retrieve Label

			# Limit Letter Count
			if len(content) > cls.max_word_count:
				content = content[:cls.max_word_count] 
			else:
				content = content.ljust(cls.max_word_count) # Pad string to max_char_in_article

			# Convert content & label to vector
			content_vector = None
			label_vector = None

			for word in content:
				temp_vec = np.array(model[word])				
				if content_vector is not None:
					content_vector = np.append(content_vector, [temp_vec], axis=0)
				else:
					content_vector = np.array([temp_vec])
			content_vector = content_vector.T

			label_eye = np.eye(cls.config.model.num_of_classes, dtype=int)
			label_vector = label_eye[int(label)]

			x_inputs.append(content_vector)
			y_labels.append(label_vector)

		return x_inputs, y_labels

	@classmethod
	def shuffleData(cls, data, label_row=3, ratio=0.6):		
		np.random.seed(randint(0,300))

		shuffle_indices = np.random.permutation(len(data)-1)

		training_indices = []
		test_indices = []

		label_0_indices = []
		label_1_indices = []
		for i in shuffle_indices:
			if (data[i][label_row] == '0'):
				label_0_indices.append(i)
			else:
				label_1_indices.append(i)

		training_size = int(len(data) * ratio)

		random.seed(2000)
		random.shuffle(label_0_indices)
		random.shuffle(label_1_indices)

		batch_size = cls.config.training.batch_size
		d1_size = int(batch_size * float(len(label_1_indices) / len(data)))
		d0_size = batch_size - d1_size
		d_indices = []
		
		i = 0
		while i*batch_size < len(data):
			d_indices.extend(label_0_indices[i*d0_size: (i+1)*d0_size])
			d_indices.extend(label_1_indices[i*d1_size: (i+1)*d1_size])
			i += 1
		print(d0_size)
		print(d1_size)
		print(len(d_indices))

		training_indices.extend(d_indices[:training_size])
		test_indices.extend(d_indices[training_size:])

		return training_indices, test_indices

	@classmethod
	def get_word_count_information_from_article_list(cls, training_dir, content_rows=[0, 1, 2]):
		np_data = CSVReader.csv_to_numpy_list(training_dir)		
		np_data = ListManipulator.merge_content_data(np_data)

		mx = -1
		mxi = -1

		mn = -1
		mni = -1

		mean = 0
		total = 0
		rows = 0

		for index, dum in enumerate(np_data):
			rows += 1
			length = len(re.compile("[\W]+").split(dum))
			total += length

			if (mx == -1) or (mx < length):
				mx = length
				mxi = index
			if (mn == -1) or (mn > length):
				mn = length
				mni = index

		mean = total / rows

		print("Maximum word count: "+str(mx))
		print("Minimum word count: "+str(mn))
		print("Average word count: "+str(mean))

