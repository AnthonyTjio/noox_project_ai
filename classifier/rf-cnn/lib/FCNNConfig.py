class ModelConfig(object):
	frame_size = 30 # From pretrained vector
	labels = "01"

	max_word_count = 800 # Sequence Length
	sequence_length = max_word_count

	num_of_classes = 2

	conv_layers = [	[256, 7, 3],
                    [256, 7, 3],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, 3]
				  ] # Based on the paper - Filter size - Pool / not - Data from pool
	num_of_filters = 256

	fully_connected_layers = [	[4608, 1024],
								[1024, 1024],
								[1024, num_of_classes]
							 ]

	rnn_sequence_length = 593
	th = 1e-6


class TrainingConfig(object):
	drop_out = 0.5
	learning_rate = 0.01
	num_of_epoches = 200

	evaluate_every = 1
	checkpoint_every = 10

	batch_size = 50
	


class HyperparametersConfig(object):
	strides = [1, 1, 1, 1]
	padding = "VALID"
	stddev = 0.1

class WordRepresentationConfig(object):
	model_name = './model'
	model_dir = './model.bin'
	learning_rate = 0.1
	dim = 30

class FCNNConfig(object):
	model = ModelConfig()
	training = TrainingConfig()
	hyper = HyperparametersConfig()
	word_model = WordRepresentationConfig()