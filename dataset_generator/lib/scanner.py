# -*- coding: utf-8 -*-

import csv
from bs4 import BeautifulSoup
from string import printable
import urllib
import numpy as np
import requests
import os
import sys 
import re
import json
from .StringManipulator import StringManipulator

class Scanner:

	filtered_words = {}

	header = {
		'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
		'Accept': 'text/html','Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
		'Accept-Encoding': 'utf-8','Connection': 'keep-alive'}

	@classmethod
	def parse(
			cls, src, target, label, 
			title_level, title_tag, title_code, 
			article_level, article_tag, article_code, error_dir='./scan_error_log.csv', src_dir=None):
		"""
			Retrieve title and article from url list and insert into csv file
			csv file will be containing src - title - article - label

			Args:
				src(str): Source name of the URL list
				target(str): Target CSV File to append
				label(int/str): Pre-determined label
				title_level(str): either h1, p, etc... used to determine title
				title_tag(str): either div or span used to determine title
				title_code(str): name of div or spam used to determine title
				article_level(str): either h1, p, etc... used to determine article
				article_tag(str): either div or span used to determine article
				article_code(str): name of div or spam used to determine article
				error_dir(str, optional): Location of error log, by default will be ./scanner_error_log.csv
				src_dir(str, optional): Location of the url list, by default will refer to ./links/ folder
		"""

		with open(target,'a') as output, open(error_dir, 'a') as error_log:
			news_writer = csv.writer(output, delimiter=',')
			error_writer = csv.writer(error_log, delimiter=",")
			link_list = []

			# Load URLs from src 
			if(src_dir is None):
				src_dir = './links/'+src+'.txt' # Default location
			# else:
				# Will load from src_dir			
			if os.path.isfile(src_dir):
				read_src = open(src_dir, 'r')
				readline_src = read_src.readlines()
				if readline_src is not None:
					for link in readline_src:
						link_list.append(link)

			for url in link_list:
				try:
					print("Processing "+url+"")
					data = []
					
					url = StringManipulator.remove_hidden_characters(url)

					r = requests.get(url)
					soup = BeautifulSoup(r.text, 'html5lib')

					if title_tag is not None:
						title = soup.find(title_level, {title_tag: re.compile(title_code)}).get_text().encode('utf-8').decode('utf-8')
					else:
						title = soup.find(title_level).get_text().encode('utf-8').decode('utf-8')

					if article_tag is not None:
						a = soup.findAll(article_level, {article_tag: re.compile(article_code)})
					else:
						a = soup.findAll(article_level)
					
					article = ""

					for i in a:
						text = i.get_text().encode('utf-8')
						article += text.decode('utf-8')

					if (article):
						print('Retrieved data from '+url)
						
						title = StringManipulator.sanitize_text_data(title)
						article = StringManipulator.sanitize_text_data(article)

						data.append(src)
						data.append(title)
						data.append(article)
						data.append(label)
						news_writer.writerow(data)
					else:
						error_message = 'Cannot find article from '+url
						data.append(src)
						data.append(url)
						data.append(error_message)

						error_writer.writerow(data)
						print(error_message)

				except Exception as ex:
					error_message = '"Exception "+str(ex)+" occured..."'
					data.append(src)
					data.append(url)
					data.append(error_message)

					error_writer.writerow(data)
					print(error_message)

	@classmethod
	def _extract_data(cls, conf_dir="./conf.json", target="./data.csv"):
		"""
			Extract source, title, and article of an url based on config file

			Args:
				conf_dir(json): Config file directory
				target(str): Destination csv data
		"""
		with open(conf_dir, 'r') as conf:
			sites = json.load(conf)
			for site in sites:
				if(site['scan']):
					src = site['src']

					title_level = site['title_level']
					title_tag = site['title_tag']
					title_code = site['title_code']

					article_level = site['article_level']
					article_tag = site['article_tag']
					article_code = site['article_code']

					label = site['label']

					cls.parse(
						src, target, label, 
						title_level, title_tag, title_code,
						article_level, article_tag, article_code)
