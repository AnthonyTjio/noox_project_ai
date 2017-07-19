import os

class WordEmbeddingConfig(object):
	model_dir = './model/word.vec'
	
class TrainingConfig(object):
	dim = 30
	lr = 0.01
	epoch = 200
	evaluate_every = 1
	min_count = 1

	word_ngrams = 5
	thread = 3
	silent = 1
	bucket = 2000000

	label_prefix ="__label__"
	label_count = 2

class ClassifierConfig(object):
	full_path = os.path.abspath("")
	model_dir = os.path.abspath('api/model/noox_stemmed_model')
	dataset_dir = os.path.abspath('api/master_train_data.txt')

class FastTextConfig(object):
	classifier = ClassifierConfig()
	word = WordEmbeddingConfig()
	training = TrainingConfig() 