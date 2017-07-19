import fasttext
import os 
import json

from .DurationRecorder import DurationRecorder
from .FastTextConfig import FastTextConfig

class FastTextProcessor:

	def __init__(self):
		self.config = FastTextConfig()
		try:
			self.classifier = fasttext.load_model(self.config.classifier.model_dir+".bin", label_prefix=self.config.training.label_prefix)
		except Exception as ex:
			self.classifier = None
			print("Model not Found")

	def train(self, training_data):
		self.classifier = fasttext.supervised(training_data, output=self.config.classifier.model_dir, dim=self.config.training.dim, 
											  lr=self.config.training.lr, epoch=self.config.training.epoch, 
											  min_count=self.config.training.min_count, word_ngrams=self.config.training.word_ngrams, 
											  thread = self.config.training.thread, silent=self.config.training.silent,
											  bucket= self.config.training.bucket)

	def predict_text(self, predict_data):
		result = self.classifier.predict_proba([predict_data], k=2)
		real_prob = float("{0:.2f}".format(result[0][0][1]))
		fake_prob = float("{0:.2f}".format(result[0][1][1]))
		prob = real_prob / (real_prob+fake_prob)
		return prob