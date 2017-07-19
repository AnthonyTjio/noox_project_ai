import os
import sys

import tensorflow as tf
import numpy as np

from lib.CCNNPreprocessor import CCNNPreprocessor
from lib.CCNNProcessor import CCNNProcessor
from lib.CSVReader import CSVReader
from lib.DurationRecorder import DurationRecorder

class Driver:
	@classmethod
	def initial_train(cls, training_dir, content_rows=[1,2], label_row=3, ratio=0.8):
		CCNNProcessor.initial_train(training_dir, content_rows=content_rows, label_row=label_row, ratio=ratio)
	
		input("Training Done, Prepare all documentation")

	@classmethod
	def plot_from_csv(cls, csv_dir, step):
		np_list = CSVReader.csv_to_numpy_list(csv_dir)
		DurationRecorder.pr_csv_plotter(np_list, every_step=step)

		input("Training Done, Prepare all documentation")
