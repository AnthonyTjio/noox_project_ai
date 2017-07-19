class ModelConfig(object):
	alphabets = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
	labels = "01"

	frame_size = len(alphabets) # Embedding Size
	embedding_size = frame_size

	max_letter_count = 2048 # Sequence Length
	sequence_length = max_letter_count

	num_of_classes = 2

	conv_layers = [	[256, 7, 3],
                    [256, 7, 3],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, None],
                    [256, 3, 3]
				  ] # Based on the paper - Filter size - Pool / not - Data from pool
	num_of_filters = 256

	num_filters_total = (max_letter_count - 96) / 27 # 96: Unknown, 27 = pools multiplication (2^3)
	num_features_total = int(num_filters_total * num_of_filters)

	fully_connected_layers = [	[16128, 1024],
								[1024, 1024],
								[1024, num_of_classes]
							 ]
	th = 1e-6


class TrainingConfig(object):
	drop_out = 0.5
	num_of_epoches = 100

	evaluate_every = 1
	checkpoint_every = 10

	batch_size = 50


class HyperparametersConfig(object):
	strides = [1, 1, 1, 1]
	padding = "VALID"
	stddev = 0.1

class CCNNConfig(object):
	model = ModelConfig()
	training = TrainingConfig()
	hyper = HyperparametersConfig()