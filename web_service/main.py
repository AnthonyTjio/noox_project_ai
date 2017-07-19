import sys
import getopt
import numpy as np

from driver import Driver

command = sys.argv[1].lower()

if(command == "createmodel"):
	opts, args = getopt.getopt(sys.argv[2:], "t:T:c:", ["train=", "test="])

	train_data = './train_data.txt'
	test_data = './test_data.txt'

	for opt, arg in opts:
		if opt in ('-t', '--train'):
			train_data = arg
		elif opt in ("-T", '--test'):
			test_data = arg

	Driver.create_model(train_data, test_data)
	
elif(command == "train"):
	opts, args = getopt.getopt(sys.argv[2:], "i:", ["input="])

	Driver.train_model()

elif(command == 'predict'):
	opts, args = getopt.getopt(sys.argv[2:], "i:f:o:", ["input=", "file=", "output="])

	texts = []
	file = None
	output = None

	for opt, arg in opts:
		if opt in ('-i', '--input'):
			texts.append(arg)
		elif opt in ('-f', '--file'):
			file = arg
		elif opt in ('-o', '--output'):
			output = arg

	if file:
		Driver.predict_txt_to_csv(file, output)
	else:
		Driver.predict_list(texts)

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