import re
import numpy as np

class ListManipulator:

	@classmethod 
	def merge_content_data(cls, np_data, content_rows=[0, 1, 2]):
		output = np.array([])

		for index, dum in enumerate(np_data):
			try:
				print("Merge #{}".format(index))
				
				data = ""
				for index in content_rows:
					data += dum[index]

				output = np.append(output, data)
			except Exception as error:
				print(dum)
				print(error)

		return output


	@classmethod
	def xor_list(cls, list_1, list_2):
		"""
			Perform xor operation on 2 list

			Args:
				list_1(list): First List
				list_2(list): Second List

			Return:
				refined_list_1(np(list)): XORed Numpy list1
				refined_list_2(np(list)): XORed Numpy list2
		"""
		xored_list = np.setxor1d(list_1, list_2)

		refined_list_1 = np.array([x for x in list_1 if x in xored_list])
		refined_list_2 = np.array([x for x in list_2 if x in xored_list])

		return refined_list_1, refined_list_2

	@classmethod
	def split_data(self, np_list, ratio):
		"""
			Split a list into 2 list randomly based on percentage

			Args:
				np_list(list): Numpy list to be splitted randomly
				ratio(int, 0 <= ratio <= 1): ratio of first product list

			Returns:
				list1(list): Splitted list which is (percentage) of source list
				list2(list): Splitted list which is (1-percentage) of source list
		"""
		if(ratio < 0 or ratio > 1):
			raise ValueError("Ratio should be between 0 and 1")

		np.random.shuffle(np_list)
		seperator = int(len(np_list)*ratio)

		list1 = np.array(np_list[:seperator]) 
		list2 = np.array(np_list[seperator:])

		return list1, list2