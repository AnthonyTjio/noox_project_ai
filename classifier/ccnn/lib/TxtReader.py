import re
import numpy as np

class TxtReader:
	@classmethod
	def read_txt_to_numpy_list(cls, src):
		data = np.array([])
		try:
			reader = open(src, 'r')
			readline_src = reader.readlines()

			if readline_src is not None:
				for text in readline_src:
					data = np.append(data, text.strip())
			else:
				raise Exception(str(src)+" is empty")

		except Exception:
			raise Exception(str(src)+" is not found")

		return data