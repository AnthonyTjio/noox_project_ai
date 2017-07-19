import csv
import os
import numpy as np
from shutil import copyfile

from .CSVReader import CSVReader
from .TxtManipulator import TxtManipulator
from .StringManipulator import StringManipulator

class FastTextPreprocessor:

	@ classmethod
	def normalize_content_data(cls, src, title, article):
		np_data = src + " " + title + " " + article
		
		np_data = StringManipulator.normalize_text(np_data, remove_stopword=True)

		return np_data
