import csv
import random
import numpy as np
import pandas as pd

class CSVReader:

	@classmethod
	def csv_to_numpy_list(self, src):
		"""
			Grabs csv file and return numpy list

			Args:
				stc(str): Csv file location

			Returns:
				np_list(np(list)): numpy containing csv data
		"""
		# with open(src, 'r') as input:
			# reader = csv.reader(input, delimiter=",")
		dat = pd.read_csv(src, delimiter=",")
		np_list = np.array(dat.as_matrix())
		return np_list

		# 	for index, row in enumerate(reader):
		# 		print("Reading index#"+ str(index))
		# 		row = np.array(row)

		# 		if(index==0):
		# 			np_list = row
		# 		else:
		# 			np_list = np.append(np_list, row, axis=0)

		# 	print("Finished Reading "+str(src))
		# 	return np_list

	@classmethod
	def csv_to_list(self, src):
		"""
			Grabs csv file and return 2 dimension list

			Args:
				src(str): csv file location
		"""
		with open(src, 'r') as input:
			reader = csv.reader(input, delimiter=",")
			data_holder = []

			for index, row in enumerate(reader):
				print("Reading index#"+str(index))
				data = []
				for column in row:
					data.append(column)

				data_holder.append(data)
				
			return data_holder

	@classmethod
	def get_csv_row(self, src, row_index):
		"""
			Grab csv row as list, use as alternative when csv_to_list exceed server memory
			
			Args:
				src(str): csv file location
				row_index(int): row to retrieve the data

			Return:
				data(list): List of data in a row
		"""
		with open(src, 'r') as input_data:
			reader = csv.reader(input_data, delimiter=",")
			data = []

			for index, row in reader:
				try:
					if index != row_index:
						continue
					for index, info in enumerate(row):
						print(info)
						data.append(info)
					return data

				except Exception as error:
					print("Not Found")
					return None

			return None

	@classmethod
	def get_csv_number_of_row(self, src):
		"""
			Retrieve number of row in csv file

			Args:
				src(str): csv file location

			Return:
				num_of_rows(int): Number of rows
		"""
		with open(src, 'r') as input:
			reader = csv.reader(input, delimiter=",")

			num_of_rows = 0
			for row in reader:
				num_of_rows += 1

			return num_of_rows

	@classmethod
	def analyze_tag_distribution(self, csv_src, label_column_index):
		"""
			Grabs csv file and analyze tag distribution

			Args:
				src(str): csv file location
				label_column_index(int): label column position in csv file
		"""
		rows = self.csv_to_list(csv_src)
		total_labels = 0
		labels = {}
		for row in rows:
			total_labels +=1
			label = row[int(label_column_index)]
			if(label not in labels):
				labels[label] = 0
			labels[label] +=1

		for label in labels.keys():
			print(str(label)+" is "+str(labels[label])+" or {0:.0f}%".format(int(labels[label])/int(total_labels) * 100))