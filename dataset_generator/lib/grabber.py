# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import re
import numpy as np
import os

class Grabber:

	header = {
		'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
		'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
		'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
		'Accept-Encoding': 'none',
		'Accept-Language': 'en-US,en;q=0.8',
		'Connection': 'keep-alive'}

	@classmethod
	def generate_links(cls, url, regex, curr_depth=None, max_depth=None):
		"""
			Generates links from an url using regex
			
			Args:
				url(str): URL to crawl
				regex(str): Regex used to filter links to grab
				curr_depth(int, optional): Used to limit number of grabs. It will produce empty
										   list if the value more than max depth. If not 
										   inserted explicitly it will grab no matter what
				max_depth(int, optional): Used to limit number of grabs. If not 
										  inserted explicitly it will grab no matter what

			Returns:
				list: List of crawled URLs
		"""
		link_list = []

		if (curr_depth is not None and max_depth is not None):
			if (curr_depth <= max_depth):
				perform = True
			else:
				perform = False
		else:
			perform = True

		if perform:
			print('Analyzing '+url)
			try:
				html_page = requests.get(url, headers= cls.header)
				soup = BeautifulSoup(html_page.text, 'html5lib')
				a = soup.findAll('a', href=re.compile(regex))
				for link in a:
					href = link['href']
					link_list.append(href)
			except Exception as error:
				print(url+' error: '+str(error)+"\n")

		return link_list

	@classmethod
	def _crawl(cls, src, regex, max_depth, temp="./links/temp.txt", base_url=None, target=None):
		"""
			Crawls through websites listed in source file and base_url and filter by regex

			Args:
				src(str): File url containing list of starting urls
				regex(str): Regex used to filter links to grab
				max_depth(int): Number of crawl iteration / crawl depth, starts from 0 (src URLs)
							until equals to depth
				temp(str, optional): Temporary file location
				base_url(str, optional): Base URL to start the crawl
				target(str, optional): File url to save product urls (Rewrite src by default)

			Returns:
				list: List of unique crawled URLs

			Produces: 
				file: Containing list of crawled URLs
		"""

		temp_access = open(temp, 'w')

		if target is None:
			target = src

		link_list = []
		link_map = {}

		# Load base_url if available
		if base_url is not None:
			link_list.append([base_url, 0])

		# Load URLs from src 
		if os.path.isfile(src):
			read_src = open(src, 'r')
			readline_src = read_src.readlines()
			if readline_src is not None:
				for link in readline_src:
					link_map[link] = True
					link_list.append([link, 0])
					temp_access.write(link+'\n')

		# Crawl using DFS
		for url, depth in link_list:
			links = cls.generate_links(url, regex, curr_depth=depth, max_depth=max_depth)
			for link in links:
				if(link not in link_map or link_map[link] is False): # If link is not yet crawled previously
					link_map[link] = True
					link_list.append([link, depth+1])
					temp_access.write(link+'\n')

		# Rotate temp with target file
		if os.path.isfile(target):
			os.remove(target)
		os.rename(temp, target)


