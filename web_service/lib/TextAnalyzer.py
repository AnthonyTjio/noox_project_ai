import nltk
from nltk.corpus import stopwords
import re
import os
import numpy as np

class TextAnalyzer:

	word_list = []
	sent_list = []

	def __init__(self, word_list_dir=None, pos_sent_list_dir=None, neg_sent_list_dir=None):
		"""
			Initializing Text Analyzer Object, User may insert the word list dictionary and sentiment 
			dictionary for further usage

			Args:
				word_list_dir(str, optional): Directory of the word dictionary list (file in txt format)
				pos_sent_dir(str, optional): Directory of positive sentiment dictionary list (file in txt format)
				neg_sent_dir(str, optional): Directory of negative sentiment dictionary list (file in txt format)
		"""
		self.word_list_dir = word_list_dir
		self.pos_sent_list_dir = pos_sent_list_dir
		self.neg_sent_list_dir = neg_sent_list_dir

		# Loading Word List Dictionary if given
		if (word_list_dir is not None):
			if os.path.isfile(word_list_dir):
				read_word = open(word_list_dir, 'r')
				readline_word = read_word.readlines()
				if readline_word is not None:
					for word in readline_src:
						self.word_list.append(word)
				else:
					raise StandardError("Word List Dictionary is empty")
			else:
				raise StandardError("Word List Dictionary is not found")

		# Loading Positive Sentiment List Dictionary if given
		if (pos_sent_list_dir is not None):
			if os.path.isfile(pos_sent_list_dir):
				read_sen = open(pos_sent_list_dir, 'r')
				readline_sen = read_sen.readlines()
				if readline_sen is not None:
					for sen in readline_src:
						self.pos_sent_list.append(sen)
				else:
					raise StandardError("Positive Sentiment List Dictionary is empty")
			else:
				raise StandardError("Positive Sentiment List Dictionary is not found")

		# Loading Negative Sentiment List Dictionary if given
		if (neg_sent_list_dir is not None):
			if os.path.isfile(neg_sent_list_dir):
				read_sen = open(neg_sent_list_dir, 'r')
				readline_sen = read_sen.readlines()
				if readline_sen is not None:
					for sen in readline_src:
						self.neg_sent_list.append(sen)
				else:
					raise StandardError("Negative Sentiment List Dictionary is empty")
			else:
				raise StandardError("Negative Sentiment List Dictionary is not found")

	def get_text_sentiment(cls, txt):
		"""
			Retrieve sentiment of a text

			Args:
				text(string): Text to be analyzed

			Returns:
				positive_sentiment(float): Positive sentiment rating
				negative_sentiment(float): Negative sentiment rating
		"""
		print("Under Construction")
		positive_sentiment = 0.0
		negative_sentiment = 0.0

		# normalize text
		# convert to freqdist
		# Get sentiment with naive bayes

		return positive_sentiment, negative_sentiment

	def retrieve_non_dictionary_words(self, text, word_list_dir=None):
		if (word_list_dir is not None):
			self.word_list_dir = word_list_dir

			# Loading Word List Dictionary if given
			if (word_list_dir is not None):
				if os.path.isfile(word_list_dir):
					read_word = open(word_list_dir, 'r')
					readline_word = read_word.readlines()
					if readline_word is not None:
						for word in readline_src:
							self.word_list.append(word)
					else:
						raise StandardError("Word List Dictionary is empty")
				else:
					raise StandardError("Word List Dictionary is not found")

		if (not self.word_list):
			return "No dictionary found"

		filtered_text = [word for word in text if word not in word_list]
		return filtered_text