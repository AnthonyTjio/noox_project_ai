import sys
import getopt
import numpy as np

from driver import Driver

command = sys.argv[1].lower()


if (command=="wordmodel"):
	opts, args =  getopt.getopt(sys.argv[2:], "i:", ["input="])

	input_file = './training.csv'

	for opt, arg in opts:
		if opt in ('-i', '--input'):
			input_file = arg

	Driver.train_word_model(input_file)

elif(command=="createmodel"):
	opts, args = getopt.getopt(sys.argv[2:], "i:d:l:r:", ["input=", "data_column=", 
															"label_column=", "ratio="])
	input_file = "./training.csv"
	data_column = []
	label_column = None
	ratio = None

	for opt, arg in opts:
		if opt in ("-i", "--input"):
			input_file = arg
		elif opt in ("-d", "--data_column"):
			data_column.append(int(arg))
		elif opt in ("-l", "--label_column"):
			label_column = int(arg)
		elif opt in ("-r", "--ratio"):
			ratio  = float(arg)

	if not data_column:
		data_column = [1,2]
	if not label_column:
		label_column = 3
	if not ratio:
		ratio = 0.8

	Driver.initial_train(input_file, content_rows=data_column, label_row=label_column, ratio=ratio)

elif(command=='count'):
	opts, args = getopt.getopt(sys.argv[2:], 'i:c:', ["input=", "column"])
	input_file = './training.csv'
	data_column = []

	for opt, arg in opts:
		if opt in ('-i', '--input'):
			input_file = arg
		elif opt in ('-c', '--column'):
			data_column.append(int(arg))

	Driver.retrieve_avg_word_count(input_file, data_column)

elif(command=="plotrecord"):
	opts, args = getopt.getopt(sys.argv[2:], "i:s:", ["input=", "step="])

	input_file = './training_log.csv'
	step = 1

	for opt, arg in opts:
		if opt in ('-i', '--input'):
			input_file = arg
		elif opt in('-s', '-step'):
			step = int(arg)

	Driver.plot_from_csv(input_file, step)