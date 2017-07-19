import sys
import getopt
import numpy as np

from driver import Driver

command = sys.argv[1].lower()

if(command == "createmodel"):
	opts, args = getopt.getopt(sys.argv[2:], "d:", ["data="])

	data = './small_train_data.txt'

	for opt, arg in opts:
		if opt in ('-d', '--data'):
			data = arg

	print("DATA: {}".format(data))
	Driver.create_model(data)

elif(command == "convert"):
	opts, args = getopt.getopt(sys.argv[2:], "i:o:d:l:p:", ["input=", "output=", "prefix=", 
															"data_column=", "label_column="])
	input_file = 'data.csv'
	output_file = 'training.txt'
	prefix = '__label__'
	label_prefix = None
	data_column = []
	label_column = 3
	
	for opt, arg in opts:
		if opt in ('-i', '--input'):
			input_file = arg
		elif opt in ('-o', 'output'):
			output_file = arg
		elif opt in ('-d', '--data_column'):
			data_column.append(int(arg))
		elif opt in ('-l', '--label_column'):
			label_column = int(arg)
		elif opt in ('-p', '--prefix'):
			label_prefix = arg

	if label_prefix:
		prefix = True

	Driver.convert_dataset(input_csv=input_file, output_txt=output_file, data_columns=data_column, 
						   label_column=label_column, label_prefix=label_prefix, 
						   append_label_prefix=prefix)
	
elif(command == "train"):
	opts, args = getopt.getopt(sys.argv[2:], "i:", ["input="])

	input_file = 'training.txt'

	for opt, arg in opts:
		if opt in ('-i', '--input'):
			input_file = arg

	Driver.train_model(input_file)

elif(command == 'test'):
	opts, args = getopt.getopt(sys.argv[2:], "i:c:", ["input="])

	input_file = 'test.txt'

	for opt, arg in opts:
		if opt in ('-i', '--input'):
			input_file = arg

	Driver.test_model(input_file)

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