import json
import os

from lib.grabber import Grabber
from lib.scanner import Scanner
from lib.similarity_checker import SimilarityChecker
from lib.CSVManipulator import CSVManipulator
from lib.CSVReader import CSVReader
from lib.TxtManipulator import TxtManipulator
from lib.TextAnalyzer import TextAnalyzer

class Driver:

	def ensure_dependencies():
		default_link_dir = "./links"
		if not os.path.exists(default_link_dir):
			os.makedirs(default_link_dir)

		default_moderation_dir = "./moderation"
		if not os.path.exists(default_moderation_dir):
			os.makedirs(default_moderation_dir)

		default_config_dir = "./conf.json"
		if not os.path.isfile(default_config_dir):
			open(default_config_dir, 'w')

		default_csv_dir = './data.csv'
		if not os.path.isfile(default_csv_dir):
			open(default_csv_dir, 'w')

	ensure_dependencies()

	@classmethod
	def crawl(cls, src='./links/', target='./links/', config='conf.json'):
		"""
			Initialize web crawler based on config

			Args:
				src(str, optional): folder directory containing initial starting points 
				target(str, optional): target folder directory to insert crawl product
				config(str, optional): config directory containing crawl details
		"""
		with open(config) as conf:
			sites = json.load(conf)
			for site in sites:
				source_directory = src+site["src"]+'.txt'
				regex = site["regex"]
				max_depth = site["iteration"]
				base_url = site["base_url"]

				if (max_depth>0):
					Grabber._crawl(source_directory, regex, max_depth, base_url=base_url)

	@classmethod
	def scan(cls, conf_dir='conf.json'):
		"""
			Retrieve title and article of URLs from text files based on config. The retrieved information will be
			inserted into data.csv by default along with the pre-determined label
		"""
		default_scanner_error_log_dir = "./scan_error_log.csv"
		open(default_scanner_error_log_dir, 'w') # Will always overwrite existing file
		
		Scanner._extract_data()

	@classmethod
	def compare(cls, csv_src="data.csv",
				src_column_index=0, title_column_index=1, article_column_index=2,
				label_column_index=3, dest_similar="./moderation/similar.csv",
				dest_unique="./moderation/unique.csv", similarity_threshold=0.75):

		SimilarityChecker.analyze(
				csv_src, src_column_index, title_column_index,
				article_column_index, label_column_index,
				dest_similar, dest_unique, similarity_threshold)

	@classmethod
	def clean_csv(cls, csv_src="./data.csv", safe=False, regex_check=False, regex="", columns_to_clean=[], 
				  threshold_check=False, threshold_column_check=[], min_legth=50, duplicate_check=False, 
				  duplicate_column_check=[]):
		
		if regex_check:
			if safe:
				CSVManipulator.safe_clean_column_with_regex(csv_src, regex, columns_to_clean)
			else:
				CSVManipulator.clean_column_with_regex(csv_src, regex, columns_to_clean)

		if threshold_check:
			CSVManipulator.remove_under_threshold_columns(csv_src, threshold_column_check, min_legth)

		if duplicate_check:
			CSVManipulator.remove_duplicate_columns(csv_src, duplicate_column_check)

	@classmethod
	def count_label(cls, csv_src, label_column_index):
		CSVReader = CSVReader()
		CSVReader.analyze_tag_distribution(csv_src, label_column_index)

	@classmethod
	def tokenize_text_file(cls, txt_src, txt_target=None, regex=None):
		TxtManipulator.tokenize_txt_file(txt_src, txt_target=txt_target, regex=regex)

	@classmethod
	def xor_text_file(cls, src_1, src_2):
		TxtManipulator.remove_similar_tokens(src_1, src_2)

	@classmethod
	def sentiment_comparison(
			cls, csv_src="data.csv", pos_sent_list_dir="./lib/sentiment_dictionary/bahasa/new_positive.txt",
			neg_sent_list_dir="./lib/sentiment_dictionary/bahasa/new_negative.txt", 
			dest_over_emotion="./moderation/overexpression.csv", 
			dest_normal_emotion="./moderation/normalexpression.csv", sentiment_threshold=0.8):

		textAnalyzer = TextAnalyzer(pos_sent_list_dir=pos_sent_list_dir, neg_sent_list_idr=neg_sent_list_dir)
		

	@classmethod
	def split_csv(cls, csv_src="data.csv", target_1="./moderation/split_1.csv", 
				  target_2="./moderation/split_2.csv", ratio=0.3):

		CSVManipulator.split_csv_data(csv_src, target_1, target_2, ratio)