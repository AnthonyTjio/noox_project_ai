import os

import numpy as np
import fasttext

from .FCNNConfig import FCNNConfig

class WordRepresentator:
	config = FCNNConfig()

	@classmethod
	def train_model(cls, input_list):
		temp_dir = 'temp_dict.txt'

		writer = open(temp_dir, 'w')

		input_list = input_list.flatten()
		print(input_list)
		for index, words in enumerate(input_list):
			print("READING #{}".format(index))
			for word in words.split(" "):			
				writer.write(word+" ")
				

		writer.close()

		model = fasttext.skipgram(temp_dir, cls.config.word_model.model_name, lr=cls.config.word_model.learning_rate, dim=cls.config.word_model.dim, min_count=3)

		os.remove(temp_dir)