import re
import string
import nltk
import numpy as np

from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

class StringManipulator:

	_stemmer = StemmerFactory().create_stemmer()
	_stop = stopwords.words('bahasa') 

	@classmethod
	def remove_non_utf(cls, text):
		"""
			Removes non utf-8 characters from string

			Args:
				text(str): source string

			Returns:
				f_text(str): Filtered string
		"""	
		f_text = ''.join(c for c in text if c in string.printable)
		return f_text

	@classmethod
	def remove_hidden_characters(cls, text):
		"""
			Removes special characters from string (Ideal for purging URLs)

			Args:
				text(str): source string

			Returns:
				f_text(str): filtered string
		"""
		f_text =  re.sub("[\s]+","", text, flags=re.IGNORECASE)
		return f_text

	@classmethod
	def remove_extra_spaces(cls, text):
		"""
			Removes extra hidden characters from string

			Args:
				text(str): source string

			Returns:
				f_text(str): filtered string
		"""
		f_text =  re.sub("\s+"," ", text, flags=re.IGNORECASE).strip()
		return f_text

	@classmethod
	def remove_infrequent_symbols_in_article(cls, text):
		"""
			Removes extra hidden characters from string

			Args:
				text(str): source string

			Returns:
				f_text(str): filtered string
		"""
		f_text =  re.sub("[^\w!?$))(-:=,.) ]+","", text)
		return f_text

	@classmethod
	def clean_using_regex(cls, text, regex):
		"""
			Clean string using regex

			Args:
				text(str): source string

			Returns:
				f_text(str): filtered string
		"""
		f_text =  re.sub(regex,"", text, flags=re.IGNORECASE).strip()
		return f_text

	@classmethod
	def custom_stopwords(cls, text, stopwords):
		"""
			Removes words in stopwords

			Args:
				text(str): source string
				stopwords(list(str)): list of stopwords

			Returns:
				f_text(str): filtered string
		"""
		text_list = text.split()
		filtered_list = [word for word in text_list if word not in stopwords]
		f_text = ' '.join(filtered_list)

	@classmethod
	def sanitize_text_data(cls, text, stopwords=None):
		"""
			Perform all sanitization to the string

			Args:
				text(str): source string

			Returns:
				f_text(str): filtered string
		"""
		text = cls.remove_non_utf(text)
		text = cls.remove_extra_spaces(text)
		text = cls.remove_infrequent_symbols_in_article(text)
		if(stopwords is not None):
			text = cls.custom_stopwords(text, stopwords)

		f_text = text
		return f_text

	@classmethod
	def remove_extra_spaces(cls, txt):
		"""
			Removes all extra spaces in a string
			
			Args:
				text(str): Source string

			Returns:
				text(str): Filtered string
		"""
		return re.sub("\s+", " ", txt).strip()

	@classmethod
	def retrieve_unique_words(cls, text):
		"""
			Returns list of unique words

			Args:
				text(str): Text string to be refactored

			Return:
				word_list(list(string)): List of unique words
		"""
		splitted_text = re.findall("([a-zA-Z]+|[0-9]+|[^\w\s]+)", text)
		word_list = np.unique(splitted_text).tolist()
		
		return word_list

	@classmethod
	def retrieve_word_count(cls, text):
		"""
			Returns list of words with its word count

			Args:
				text(str): Text string to be refactored

			Return:
				word_list(dict{string: qty}): List of unique words and its count
		"""
		word_list = nltk.FreqDist(text.lower().split())
		return word_list

	@classmethod
	def normalize_text(cls, text):
		"""
			Returns normalized version of the text
			Normalization means removal of non-alphanumeric characters and stemming of word.
			Also remove stopwords

			Args:
				text(str): Text to be normalized

			Return: 
				normalized_text(str): Normalized text
		"""
		normalized_list = cls._stemmer.stem(text)
		normalized_list = [word for word in normalized_list.split(" ") if word not in cls._stop]
		normalized_text = " ".join(normalized_list)

		return normalized_text