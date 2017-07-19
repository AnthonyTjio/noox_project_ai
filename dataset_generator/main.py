# -*- coding: utf-8 -*-
import sys
import getopt
from driver import Driver

if (len(sys.argv) > 1):
	command = sys.argv[1].lower()
else:
	readme_src= open("readme.md", 'r')
	readline_src = readme_src.readlines()
	for line in readline_src:
		print(line)

	sys.exit()

if(command == "crawl"):
	Driver.crawl()

elif(command == "scan"):
	Driver.scan()

elif(command == "compare"):
	opts, args = getopt.getopt(sys.argv[2:], "hs:", ["help=", "source="])
	
	src = None
	
	for opt, arg in opts:
		if opt in ("-s", "source"):
			src = arg
		elif opt in ("-h", "--help"):
			print("main.py -s <SafeStatus> -s <csv_source>")
			sys.exit()

	if (src is None):
		src = "./data.csv"

	Driver.compare(csv_src=src)

elif(command == "clean"):
	opts, args = getopt.getopt(sys.argv[2:], "hS:s:R:r:D:d:T:t:", 
											["help", "safe=", "source=", "regex=", "regex_column=", "duplicate=",
											 "duplicate_column=", "threshold=", "threshold_column="])
	if (len(sys.argv) >=5):
		safe = False
		src = None
		regex_check = False
		regex = ""
		regex_column =[]
		threshold_check = False
		threshold = 0
		threshold_column = []
		duplicate_check = False
		duplicate_column = []
		for opt, arg in opts:
			if opt in ("-S", "--safe"):
				safe = arg
			elif opt in ("-s", "--source"):
				src = arg
			elif opt in ("-R", "--regex"):
				regex = arg
				regex_check = True
			elif opt in ("-r", "--regex_column"):
				regex_column.append(int(arg))
			elif opt in ("-D", "--duplicate"):
				duplicate_check = arg
			elif opt in ("-d", "--duplicate_column"):
				duplicate_column.append(int(arg))
			elif opt in ("-T", "--threshold"):
				threshold = arg
				threshold_check = True
			elif opt in ("-t", "--threshold_column"):
				print(arg)
				threshold_column.append(int(arg))
			elif opt in ("-h", "--help"):
				print("main.py -S <SafeStatus> -s <csv_source> -r <regex> -c <column_num> -c <column_num>",
					  "-D <DuplicateStatus> -d <duplicate_column_num> -d <duplicate_column_num>",
					  "-T <NumOfCharMinThreshold> -t <threshold_column_num> -t <threshold_column_num>")
				sys.exit()
		if (src is None):
			src = "./data.csv"

		Driver.clean_csv(src, safe=safe, regex_check=regex_check, regex=regex, columns_to_clean=regex_column,
						 threshold_check=threshold_check, threshold_column_check=threshold_column,
						 min_legth=threshold, duplicate_check=duplicate_check, 
						 duplicate_column_check=duplicate_column)
	else:
		print("main.py -S <SafeStatus> -s <csv_source> -r <regex> -c <column_num> -c <column_num>",
					  "-D <DuplicateStatus> -d <duplicate_column_num> -d <duplicate_column_num>",
					  "-T <NumOfCharMinThreshold> -t <threshold_column_num> -t <threshold_column_num>")
		sys.exit()

elif(command == "count"):
	opts, args = getopt.getopt(sys.argv[2:], "s:c:h", ["source=", "column=", "help"])
	src = "./data.csv"
	column = 3

	for opt, arg in opts:
		if opt in ["-s", "--source"]:
			src = arg
		elif opt in ("-c", "--column"):
			src = int(arg)
		elif opt in ("-h", "--help"):
			print("main.py -s <sourcefile> -c <columnindex>")

	Driver.count_label(src, column)

elif(command == "tokentxtfile"):
	opts, args = getopt.getopt(sys.argv[2:], "i:o:r:", ["input=", "output=", "regex="])

	input_file = None
	output_file = None
	regex = None

	for opt, arg in opts:
		if opt in ("-i", "--input"):
			input_file = arg
		elif opt in ("-o", "--output"):
			output_file = arg
		elif opt in("-r", "--regex"):
			regex = arg

	if input_file == None:
		print("main.py tokentxtfile -i <inputfile> -o <outputfile> -r <regex>")
	else:
		Driver.tokenize_text_file(txt_src=input_file, txt_target=output_file, regex=regex)

elif(command == "xortoken"):
	if(len(sys.argv)==4):
		src_1_dir = sys.argv[2]
		src_2_dir = sys.argv[3]
		Driver.xor_text_file(src_1_dir, src_2_dir)
	else:
		print("main.py xortoken <src_1_dir> <src_2_dir>")

elif(command == "splitcsv"):
	if(len(sys.argv)==6):
		src_dir = sys.argv[2]
		target_1_dir = sys.argv[3]
		target_2_dir = sys.argv[4]
		ratio = float(sys.argv[5])
		Driver.split_csv(src_dir, target_1_dir, target_2_dir, ratio)
	else:
		print("main.py splitcsv <src_dir> <target_1_dir> <target_2_dir> <ratio>")