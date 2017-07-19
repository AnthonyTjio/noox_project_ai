import re
import nltk
import numpy as np
import os
import time

from .StringManipulator import StringManipulator
from .ListManipulator import ListManipulator
from .StringManipulator import StringManipulator
from .FileManipulator import FileManipulator
from .TxtReader import TxtReader

class TxtManipulator:

	@classmethod
	def tokenize_txt_file(cls, txt_src, txt_target=None, regex=None):
		"""
			Converts txt file into tokens 

			Args:
				txt_src(str): Source txt file location
				txt_target(str, optional): Target txt file location, if left blank will replace the src file
				regex(str, optional): Regex on how to split the tokens, if left blank will use space 
		"""
		text_list = []

		if not regex:
			regex = "\s+"

		if not txt_target:
			txt_target = txt_src[:-4]+"_tokenized.txt"

		temp_target = txt_target[:-4]+"_temp.txt"
		txt_writer = open(temp_target, 'w')
		word_count = 0

		try:
			read_src = open(txt_src, 'r')
			readline_src = read_src.readlines()
			if readline_src is not None:
				for text in readline_src:
					text = StringManipulator.normalize_text(text)
					text = re.compile(regex).split(text)
					for word in text:
						word_count += 1 
						text_list.append(word.strip())

			else:
				raise Exception('Source not found error')
			
			word_list = np.unique(text_list).tolist()

			for index, word in enumerate(word_list):
				if(word):
					txt_writer.write(word+"\n")

			txt_writer.close()
			FileManipulator.rotate_file(txt_target, temp_target)

		except Exception as error:
			print(str(error)+" has ocurred...")


	@classmethod
	def remove_similar_tokens(cls, src_1_dir, src_2_dir):
		"""
			Removes similar tokens from 2 txt sources

			Args:
				src1(str): First txt source
				src2(str): Second txt source
		"""
		temp_src_1_dir = src_1_dir[:-4]+"_temp.txt"
		temp_src_2_dir = src_2_dir[:-4]+"_temp.txt"

		try:
			src_1_list = TxtReader.read_txt_to_numpy_list(src_1_dir)
			src_2_list = TxtReader.read_txt_to_numpy_list(src_2_dir)

			src_1_list, src_2_list = ListManipulator.xor_list(src_1_list, src_2_list)

			cls.write_txt_data_from_1d_np_list(temp_src_1_dir, src_1_list)
			cls.write_txt_data_from_1d_np_list(temp_src_2_dir, src_2_list)

			FileManipulator.rotate_file(src_1_dir, temp_src_1_dir)
			FileManipulator.rotate_file(src_2_dir, temp_src_2_dir)

		except Exception as error:
			print(str(error)+" has ocurred...")

	@classmethod
	def write_txt_data_from_1d_np_list(cls, target_dir, np_list):
		"""
			Writes np list data to txt file

			Args:
				target_dir(str): txt to be writtern
				np_list(np(list): 1d numpy to be written into csv
		"""
		writer = open(target_dir, 'w')

		for word in np_list:
			writer.write(word.strip()+"\n")

		writer.close()

	@classmethod
	def append_txt_data_from_1d_np_list(cls, target_dir, np_list):
		"""
			Appends np list data to txt file

			Args:
				target_dir(str): txt to be writtern
				np_list(np(list): 1d numpy to be written into csv
		"""
		writer = open(target_dir, 'a')

		for word in np_list:
			writer.write(word.strip()+"\n")

		writer.close()
