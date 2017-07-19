import nltk
import numpy as np

from .StringManipulator import StringManipulator

class TFIDFCalculator:

	def __init__(self):
		self.word_list = np.array([])

	def load_possible_terms(self, np_text_list):
		"""
			Retrieve possible words/terms from numpy list of text

			Args:
				np_text_list(np(list(string))): Numpy list containing text which term to be extracted
		"""

		temp_word_list = np.array([])

		for text in np_text_list:
			text = StringManipulator.normalize_text(text)
			temp_word_list = np.append(temp_word_list, StringManipulator.retrieve_unique_words(text))

		self.word_list = np.append(self.word_list, temp_word_list)
		self.word_list = np.unique(self.word_list)

		self.text_collection = nltk.TextCollection(self.word_list)

	def calculate_tf_idf(self, np_text_list):
		"""
			Calculate tf-idf of np list of text
			TF: Num of times word appears in a text list/ Total num of words in that text

			Args:
				np_text_list(np(list(str))): Numpy list of text to calculate tf-idf

			Return:
				tf_dict(dict(index: (dict(term: tf_idf)))): dictionary of text's index where it contains dictionary
															of terms containing freq dist
		"""
		tf_idf = {}

		#get tf_idf score
		for index, text in enumerate(text_list):
			text = StringManipulator.normalize_text(text)
			text_dist = StringManipulator.retrieve_word_count(text)
			tf_idf[index] = {}
			for term in text_dist.keys():
				tf_idf[index][term] = self.text_collection.tf_idf(term, text)

		return tf_idf