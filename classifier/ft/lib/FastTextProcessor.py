import fasttext
import os 
import json

from sklearn.cross_validation import KFold

from .DurationRecorder import DurationRecorder
from .FastTextConfig import FastTextConfig
from .TxtReader import TxtReader
from .TxtManipulator import TxtManipulator

class FastTextProcessor:

	def __init__(self):
		self.config = FastTextConfig()
		try:
			self.classifier = fasttext.load_model(self.config.classifier.model_dir+".bin", label_prefix=self.config.training.label_prefix)
		except Exception as ex:
			self.classifier = None
			print("Model not Found")

	def initial_train(self, data_dir):			
		input_data = TxtReader.read_txt_to_numpy_list(data_dir)
		kf = KFold(len(input_data), n_folds=10, shuffle=True)

		training_data = './temp_train_data.txt'
		test_data = './temp_test_data.txt'

		index = 0

		for train_indices, test_indices in kf:
			index += 1
			# Create temp training & test data
			TxtManipulator.write_txt_data_from_1d_np_list(training_data, input_data[train_indices])
			TxtManipulator.write_txt_data_from_1d_np_list(test_data, input_data[test_indices])

			DurationRecorder.start_log()			

			self.classifier = fasttext.supervised(training_data, output=self.config.classifier.model_dir, dim=self.config.training.dim, 
											  lr=self.config.training.lr, epoch=self.config.training.epoch, 
											  min_count=self.config.training.min_count, word_ngrams=self.config.training.word_ngrams, 
											  thread = self.config.training.thread, silent=self.config.training.silent,
											  bucket= self.config.training.bucket)
			# Increasing epoch

			result = self.classifier.test(test_data)		

			precision = result.precision
			recall = result.recall
			DurationRecorder.pr_epoch_logger(index, precision, recall)
			DurationRecorder.pr_epoch_plotter(index, precision, recall)

	def train(self, training_data):
		self.classifier = fasttext.supervised(training_data, output=self.config.classifier.model_dir, dim=self.config.training.dim, 
											  lr=self.config.training.lr, epoch=step, 
											  min_count=self.config.training.min_count, word_ngrams=self.config.training.word_ngrams, 
											  thread = self.config.training.thread, silent=self.config.training.silent,
											  bucket= self.config.training.bucket)

	def test(self, test_data):
		result = self.classifier.test(test_data)
		return result

	def predict(self, predict_data):
		result = self.classifier.predict_proba(predict_data, k=2)
		return result