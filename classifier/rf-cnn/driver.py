import os
import sys

import tensorflow as tf
import numpy as np

from lib.CSVReader import CSVReader
from lib.FCNNPreprocessor import FCNNPreprocessor
from lib.FCNNProcessor import FCNNProcessor
from lib.WordRepresentator import WordRepresentator
from lib.DurationRecorder import DurationRecorder



class Driver:
	@classmethod
	def train_word_model(cls, training_dir, content_rows=[0, 1,2]):
		data = CSVReader.csv_to_numpy_list(training_dir) # 2D-Array (Elements, Variables)
		data = FCNNPreprocessor.normalize_content_data(data)
		text_data = FCNNPreprocessor.get_content_data(data, content_rows)
		word_model = WordRepresentator.train_model(text_data)

	@classmethod
	def initial_train(cls, training_dir, content_rows=[0, 1, 2], label_row=3, ratio=0.8):
		data = CSVReader.csv_to_numpy_list(training_dir) # 2D-Array (Elements, Variables)
		data = FCNNPreprocessor.normalize_content_data(data)
		text_data = FCNNPreprocessor.get_content_data(data, content_rows)
		word_model = WordRepresentator.train_model(text_data)

		FCNNProcessor.initial_train(training_dir, content_rows=content_rows, label_row=label_row, ratio=ratio)
	
	@classmethod
	def retrieve_avg_word_count(cls, csv_dir, content_rows=[0,1,2]):
		FCNNPreprocessor.get_word_count_information_from_article_list(csv_dir, content_rows)

	@classmethod
	def plot_from_csv(cls, csv_dir, step):
		np_list = CSVReader.csv_to_numpy_list(csv_dir)
		DurationRecorder.pr_csv_plotter(np_list, every_step=step)

		input("Training Done, Prepare all documentation")