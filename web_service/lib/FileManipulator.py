import numpy as np
import csv
import os
from shutil import copyfile

class FileManipulator:

	@classmethod
	def rotate_file(cls, main_dir, temp_dir):
		"""
			Rotate temp file with main file

			Args:
				main_dir(str): Location of main file
				temp_dir(str): Location of temporary file
		"""
		if(os.path.isfile(main_dir)):
			os.remove(main_dir)
		copyfile(temp_dir, main_dir)
		os.remove(temp_dir)