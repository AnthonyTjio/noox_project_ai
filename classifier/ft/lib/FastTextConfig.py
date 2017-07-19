class WordEmbeddingConfig(object):
	model_dir = './model/word.vec'
	
class TrainingConfig(object):
	dim = 90
	lr = 1e-2
	epoch = 200
	evaluate_every = 1
	min_count = 1

	word_ngrams = 3
	thread = 3
	silent = 1
	bucket = 2000000

	label_prefix ="__label__"
	label_count = 2

class ClassifierConfig(object):
	model_dir = './model/noox_model'

class FastTextConfig(object):
	classifier = ClassifierConfig()
	word = WordEmbeddingConfig()
	training = TrainingConfig() 