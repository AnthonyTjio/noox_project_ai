import fasttext
import json
import os
import numpy as np

from .lib.FastTextPreprocessor import FastTextPreprocessor
from .lib.FastTextProcessor import FastTextProcessor
from .lib.FastTextConfig import FastTextConfig

from .lib.CSVManipulator import CSVManipulator
from .lib.TxtReader import TxtReader
from .lib.TxtManipulator import TxtManipulator
from .lib.CSVReader import CSVReader
from .lib.DurationRecorder import DurationRecorder
from .lib.StringManipulator import StringManipulator

class Driver:
	def ensure_dependencies():
		default_model_dir = "./model"
		if not os.path.exists(default_model_dir):
			os.makedirs(default_model_dir)

	ensure_dependencies()

	@classmethod
	def convert_dataset(cls, input_csv, output_txt, data_columns, 
									  label_column, label_prefix="__label__", 
									  append_label_prefix=False):
		FastTextPreprocessor.convert_csv_to_fasttext_input(input_csv, output_txt, data_columns, 
														   label_column, label_prefix, append_label_prefix)

	@classmethod
	def train_model(cls):
		fasttextProcessor = FastTextProcessor()
		dataset_dir = FastTextConfig.classifier.dataset_dir

		fasttextProcessor.train(dataset_dir)

	@classmethod
	def insert_dataset(cls, src, title, article, label):
		label_input = "\n"+FastTextConfig.training.label_prefix+str(label)
		
		np_text_input = FastTextPreprocessor.normalize_content_data(src, title, article)
		text_input = ' '.join(str(word) for word in np_text_input)
		text_input = StringManipulator.remove_extra_spaces(text_input)

		dataset = [label_input + ", " + text_input]
		dataset_dir = FastTextConfig.classifier.dataset_dir

		TxtManipulator.append_txt_data_from_1d_np_list(dataset_dir, dataset)

	@classmethod
	def predict_text(cls, src, title, article):
		fasttextProcessor = FastTextProcessor()

		np_text_input = FastTextPreprocessor.normalize_content_data(src, title, article)
		text_input = StringManipulator.remove_extra_spaces(np_text_input)

		real_prob, fake_prob = fasttextProcessor.predict_text(text_input)

		return real_prob, fake_prob


