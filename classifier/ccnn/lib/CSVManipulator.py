import csv
import re
import os
import numpy as np
import pandas as pd
from shutil import copyfile

from .CSVReader import CSVReader
from .FileManipulator import FileManipulator
from .StringManipulator import StringManipulator
from .ListManipulator import ListManipulator

class CSVManipulator:
	@classmethod
	def clean_column_with_regex(cls, csv_dir, regex, columns_to_clean=None, clean_extra_spaces=True):
		"""
			Remove matching regex from csv file column

			Args:
				csv_dir(str): Location of csv file
				regex(str): Regex to be removed from the column
				column_to_clean(list(int), optional): CSV column to be cleaned, if left blank will clean whole column
				clean_extra_spaces(bool, Optional): Should it clean extra spaces as well?
		"""
		temp_dir = csv_dir[:-4]+"_temp.csv"

		rows = CSVReader.csv_to_list(csv_dir)
		if (columns_to_clean is None):
			columns_to_clean = range(len(rows[0]))

		with open(temp_dir, "w") as output:
			csv_writer = csv.writer(output, delimiter=",")
			for row in rows:
				for i in columns_to_clean:
					i = int(i)
					row[i] = re.sub(regex, "", row[i])
					row[i] = StringManipulator.remove_extra_spaces(row[i])

				data = []
				for info in row:
					data.append(info)

				csv_writer.writerow(data)

		FileManipulator.rotate_file(csv_dir, temp_dir)

	@classmethod
	def safe_clean_column_with_regex(cls, csv_src, regex, columns_to_clean=None, clean_extra_spaces=True):
		"""
			Remove matching regex from csv file column safely, use as alternative when csv_to_list throw memory error

			Args:
				csv_src(str): Location of csv file
				regex(str): Regex to be removed from the column
				column_to_clean(list(int), optional): CSV column to be cleaned, if left blank will clean whole column
				clean_extra_spaces(bool, Optional): Should it clean extra spaces as well?
		"""
		temp_dir = csv_src[:-4]+"_temp.csv"

		row_count = CSVReader.get_csv_number_of_row(csv_src)

		with open(csv_src, 'r') as input_data, open(temp_dir, "w") as output_data:
			csv_reader = csv.reader(input_data, delimiter=',')
			csv_writer = csv.writer(output_data, delimiter=",")

			for row_index, row in enumerate(csv_reader):
				data = []
				print("Cleaning index #"+str(row_index))

				for column_index, column in enumerate(row):
					if column_index in columns_to_clean or not columns_to_clean:
						column_data = re.sub(regex, "", column)
						data.append(column_data)
					else:
						data.append(column)

				csv_writer.writerow(data)

		cls.rotate_file(csv_src, temp_dir)

	@classmethod
	def remove_under_threshold_columns(cls, csv_src, columns_to_clean, min_length):
		"""
			Remove rows which column length is lower than minimum length

			Args:
				csv_src(str): Location of csv file
				columns_to_clean(list(int)): List of columns to be clean, if left blank will check whole column
				min_length(int): Length threshold of allowable column length
		"""
		temp_dir = csv_src[:-4]+"_temp.csv"
		min_length = int(min_length)

		row_count = CSVReader.get_csv_number_of_row(csv_src)

		with open(csv_src, 'r') as input_data, open(temp_dir, "w") as output_data:
			csv_reader = csv.reader(input_data, delimiter=',')
			csv_writer = csv.writer(output_data, delimiter=",")

			for row_index, row in enumerate(csv_reader):
				data = []
				ok = True
				print("Checking threshold on index #"+str(row_index))
				for column_index, column in enumerate(row):
					try:
						if column_index in columns_to_clean or not columns_to_clean:
							if (len(column) < min_length):
								ok = False
								break
							else:
								data.append(column)
						else:
							data.append(column)
					except Exception as error:
						print(str(error)+" has occured...")

				if (ok):
					csv_writer.writerow(data)

		FileManipulator.rotate_file(csv_src, temp_dir)

	@classmethod
	def remove_duplicate_columns(cls, csv_src, columns_to_clean):
		"""
			Remove row which column is duplicated with other row

			Args:
				csv_src(str): Location of csv file
				columns_to_clean(list(int)): List of columns to be clean, if left blank will check whole column
		"""
		temp_dir = csv_src[:-4]+"_temp.csv"

		row_count = CSVReader.get_csv_number_of_row(csv_src)

		with open(csv_src, 'r') as input_data, open(csv_src, 'r') as temp_data, \
			 open(temp_dir, "w") as output_data:

			csv_reader = csv.reader(input_data, delimiter=',')
			temp_csv_reader = csv.reader(temp_data, delimiter=',')
			csv_writer = csv.writer(output_data, delimiter=",")

			previous_index = -1
			blacklisted_indexes = []
			for main_row_index, main_row in enumerate(csv_reader):
				if (main_row_index == previous_index or main_row_index in blacklisted_indexes):
					continue

				data = []
				print("Cleaning duplicates of index #"+str(main_row_index))

				for main_column_index, main_column in enumerate(main_row):
					data.append(main_column)

				csv_writer.writerow(data)

				temp_data.seek(0)
				for secondary_row_index, secondary_row in enumerate(temp_csv_reader):
					if(secondary_row_index <= main_row_index or secondary_row_index in blacklisted_indexes):
						continue

					for secondary_column_index, secondary_column in enumerate(secondary_row):
						if (secondary_column_index in columns_to_clean or not columns_to_clean):
							try:
								text1 = data[secondary_column_index]
								text2 = secondary_column
								if(text1==text2):
									print("Similar Index: "+str(secondary_row_index))
									blacklisted_indexes.append(secondary_row_index)
									break
							except Exception as error:
								print(str(error)+" has occured")			
				
		FileManipulator.rotate_file(csv_src, temp_dir)

	@classmethod
	def split_csv_data(cls, csv_src, csv_target_dir_1, csv_target_dir_2, ratio):
		np_data = CSVReader.csv_to_numpy_list(csv_src)
		np_data_1, np_data_2 = ListManipulator.split_data(np_data, ratio)

		temp_target_1 = csv_target_dir_1[:-4]+"_temp.csv"
		temp_target_2 = csv_target_dir_2[:-4]+"_temp.csv"

		cls.write_csv_data_from_2d_np_list(temp_target_1, np_data_1)
		cls.write_csv_data_from_2d_np_list(temp_target_2, np_data_2)

		FileManipulator.rotate_file(csv_target_dir_1, temp_target_1)
		FileManipulator.rotate_file(csv_target_dir_2, temp_target_2)

	@classmethod
	def write_csv_data_from_2d_np_list(cls, target_dir, np_list):
		"""
			Writes np list data to csv

			Args:
				target_dir(str): csv to be writtern
				np_list(np(list(list))): 2d numpy to be written into csv
		"""
		with open(target_dir, 'w') as output:
			csv_writer = csv.writer(output, delimiter=",")

			for row in np_list:
				data = []
				
				for column in row:
					data.append(column)

				csv_writer.writerow(data)
				del data

	@classmethod
	def append_csv_data_from_2d_np_list(cls, target_dir, np_list):
		"""
			Appends np list data to csv

			Args:
				target_dir(str): csv to be writtern
				np_list(np(list(list))): 2d numpy to be written into csv
		"""
		with open(target_dir, 'a') as output:
			csv_writer = csv.writer(output, delimiter=",")

			for row in np_list:
				data = []
				
				for column in row:
					data.append(column)

				csv_writer.writerow(data)


