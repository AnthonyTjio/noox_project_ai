import csv
import os
import numpy as np
from shutil import copyfile

from .CSVReader import CSVReader
from .TxtManipulator import TxtManipulator
from .StringManipulator import StringManipulator

class FastTextPreprocessor:

	@classmethod
	def convert_csv_to_fasttext_input(cls, input_csv, output_txt, data_columns, 
									  label_column, label_prefix="__label__", 
									  append_label_prefix=False, stemming=False, remove_stopword=False):
		file = CSVReader.csv_to_numpy_list(input_csv)
		file = cls.normalize_content_data(file, stemming=stemming, remove_stopword=remove_stopword)
		
		data = 	file[:,data_columns]
		if append_label_prefix:
			label_column = np.array([label_prefix+str(row[label_column]) for row in file])
		else:
			label_column = file[:, label_column]

		output_data = np.array([str(label_column[index])+' , '+' '.join(str(col) for col in row) 
					  for index, row in enumerate(data)])
		TxtManipulator.write_txt_data_from_1d_np_list(output_txt, output_data)

	@classmethod
	def normalize_content_data(cls, np_data, stemming=False, remove_stopword=False):
		if stemming:
			for i, dum in enumerate(np_data):
				print("Normalizing #{} data".format(i))
				for j, dum in enumerate(dum):
					np_data[i,j] = str(np_data[i,j])
					np_data[i,j] = StringManipulator.normalize_text(np_data[i,j], remove_stopword=remove_stopword)
		else:
			for i, dum in enumerate(np_data):
				print("Normalizing #{} data".format(i))
				for j, dum in enumerate(dum):
					np_data[i,j] = str(np_data[i,j])
					np_data[i,j] = np_data[i,j].lower()
					if remove_stopword:
						np_data[i,j] = StringManipulator.remove_stopwords(np_data[i,j])
	
		return np_data