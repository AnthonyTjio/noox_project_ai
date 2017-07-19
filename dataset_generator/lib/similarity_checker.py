import nltk
import numpy as np
import csv
from nltk.corpus import stopwords

from .StringManipulator import StringManipulator
from .CSVReader import CSVReader
from .TFIDFCalculator import TFIDFCalculator

class SimilarityChecker:

	@classmethod
	def analyze(
			cls, csv_src, src_column_index, title_column_index, article_column_index, label_column_index,
		 	dest_similar, dest_unique, similarity_threshold):
		"""
			Retrieve news information in csv file and compare similar articles from different sources

			Args:
				csv_src(str): Source CSV directory
				src_column_index(int): Column index of source
				title_column_index(int): Column index of title
				article_column_index(int): Column index of content
				label_column_index(int): Column index of label
				dest_similar(str): Destination CSV file which contains similar articles
				dest_unique(str): Destination CSV file which only contain unqiue articles
				similarity_threshold(float[0 < x < 1]) similarity limit to be considered as similar (1 is exactly similar)
		"""
		with open(dest_similar, 'a') as sim_output, open(dest_unique, 'a') as uni_output:
			sim_writer = csv.writer(sim_output, delimiter=",")
			uni_writer = csv.writer(uni_output, delimiter=",")

			TfIdfCalculator = TFIDFCalculator()

			similarity_threshold = 1-similarity_threshold
			np_news = CSVReader.csv_to_numpy_list(csv_src)

			text_list = np_news[:,article_column_index].tolist() # Retrieve articles to list
			np_text_list = np_news[:, article_column_index]

			TfIdfCalculator.load_possible_terms(text_list)
			tf_idf = TfIdfCalculator.calculate_tf_idf(np_text_list)

			classified_index = []
			similar_index = 0

			for main_index, main_news in enumerate(np_news):
				if (main_index in classified_index):
					continue

				print("Comparing "+main_news[title_column_index]+" from "+main_news[src_column_index])

				classified_index.append(main_index)
				similar_indexes = []

				for secondary_index in range(main_index, len(np_news)):
					if (secondary_index in classified_index):
						continue

					secondary_news = np_news[secondary_index]
					term1 = tf_idf[main_index].copy()
					term2 = tf_idf[secondary_index].copy()

					for term in term1:
						if term not in term2:
							term2[term] = 0

					for term in term2:
						if term not in term1:
							term1[term] = 0

					v1 = [score for (term, score) in sorted(term1.items())]
					v2 = [score for (term, score) in sorted(term2.items())]

					distance = nltk.cluster.util.cosine_distance(v1,v2) # closer to 0 is more similar

					if(distance < similarity_threshold):
						similar_indexes.append(secondary_index)

				if similar_indexes:
					similar_index += 1
					data = []
					data.append(similar_index)
					data.append(main_news[src_column_index])
					data.append(main_news[title_column_index])
					data.append(main_news[article_column_index])
					data.append(main_news[label_column_index])
					sim_writer.writerow(data)

					for index in similar_indexes:
						data = []
						secondary_news = np_news[index]
						classified_index.append(index)

						data.append(similar_index)
						data.append(secondary_news[src_column_index])
						data.append(secondary_news[title_column_index])
						data.append(secondary_news[article_column_index])
						data.append(secondary_news[label_column_index])
						sim_writer.writerow(data)
				else:
					data = []
					data.append(main_news[src_column_index])
					data.append(main_news[title_column_index])
					data.append(main_news[article_column_index])
					data.append(main_news[label_column_index])
					uni_writer.writerow(data)