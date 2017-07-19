import json
import os
import time
import datetime

import sklearn
import sklearn.metrics
import tensorflow as tf
import numpy as np

from .CSVReader import CSVReader
from .FCNNModel import FCNNModel
from .FCNNPreprocessor import FCNNPreprocessor
from .FCNNConfig import FCNNConfig
from .ListManipulator import ListManipulator
from .DurationRecorder import DurationRecorder

class FCNNProcessor:

	config = FCNNConfig()

	@classmethod
	def train_step(cls, x_batch, y_batch, cnn, session, train_op, global_step, train_summary_op, 
	 				train_summary_writer):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: cls.config.training.drop_out
            }

            _, step, summaries, loss, accuracy, predictions= session.run(
                [train_op, # Updates parameter of the network
                 global_step, 
                 train_summary_op, 
                 cnn.loss, 
                 cnn.accuracy,
                 cnn.predictions],
                feed_dict)

            batch_labels = np.argmax(y_batch, 1)
            precision = sklearn.metrics.precision_score(batch_labels, predictions)
            recall = sklearn.metrics.recall_score(batch_labels, predictions)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, acc {}, precision {}, recall: {}".format(time_str, step, loss, 
            																	  accuracy, precision, recall))
            train_summary_writer.add_summary(summaries, step)

            return time_str, step, loss, accuracy, precision, recall

	@classmethod
	def dev_step(cls, x_batch, y_batch, cnn, session, global_step, dev_summary_op, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0 # Disables learning
            }

            step, summaries, loss, accuracy, predictions = session.run(
                [global_step, 
                 dev_summary_op, 
                 cnn.loss, 
                 cnn.accuracy,
                 cnn.predictions],
                feed_dict)

            batch_labels = np.argmax(y_batch, 1)
            precision = sklearn.metrics.precision_score(batch_labels, predictions)
            recall = sklearn.metrics.recall_score(batch_labels, predictions)     

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, acc {}, precision {}, recall: {}".format(time_str, step, loss, 
            																	  accuracy, precision, recall))
            if writer:
            	writer.add_summary(summaries, step)

            return time_str, step, loss, accuracy, precision, recall	

	@classmethod
	def initial_train(cls, training_dir, content_rows=[1,2], label_row=3, ratio=0.8):
		with tf.Graph().as_default():
			session_conf = tf.ConfigProto(
				allow_soft_placement=True,
				log_device_placement=False) #Let's tweak this later

			session = tf.Session(config=session_conf)
			
			with session.as_default():
				cnn = FCNNModel()

				# Define Training Procedure
				global_step = tf.Variable(0, name="global_step", trainable="false")
				optimizer = tf.train.AdamOptimizer(cls.config.model.th)
				grads_and_vars = optimizer.compute_gradients(cnn.loss)
				train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

			# Summarizing
			grad_summaries = []
			for g, v in grads_and_vars:
				if g is not None:
					grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
					sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
					grad_summaries.append(grad_hist_summary)
					grad_summaries.append(sparsity_summary)
			grad_summaries_merged = tf.summary.merge(grad_summaries)

			# Output Directory for models & summaries
			timestamp = str(int(time.time()))
			out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
			print('Writing to {}\n'.format(out_dir))

			# Summaries for loss and accuracy
			loss_summary = tf.summary.scalar("loss", cnn.loss)
			acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

			# Train Summaries - Ensure train summary dir exist & fill
			train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
			train_summary_dir =  os.path.join(out_dir, "summaries", "train")
			train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph)

			# Dev Summaries - Ensure dev summary exist & fill
			dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
			dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
			dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph)

			# Checkpoint Directory - Ensure checkpoint (model) directory exist
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)
			saver = tf.train.Saver(tf.global_variables())

			# Initialize All Variables
			session.run(tf.global_variables_initializer())
			curr_time = None

			print("Reading Data")
			data = CSVReader.csv_to_numpy_list(training_dir) # 2D-Array (Elements, Variables)
			print("Finished Reading")

			data = FCNNPreprocessor.normalize_content_data(data)
			input_data, label_data = FCNNPreprocessor.convert_dataset(data)

			training_indices, test_indices = FCNNPreprocessor.shuffleData(data, ratio=ratio)

			training_input_data = np.array(input_data[training_indices])
			training_label_data = np.array(label_data[training_indices])

			test_input_data = np.array(input_data[test_indices])
			test_label_data = np.array(label_data[test_indices])

			training_input_data = np.squeeze(training_input_data, axis=1)
			training_label_data = np.squeeze(training_label_data, axis=1)

			test_input_data = np.squeeze(test_input_data, axis=1)
			test_label_data = np.squeeze(test_label_data, axis=1)

			DurationRecorder.start_log()

			for epoch in range(cls.config.training.num_of_epoches):
				print("Processing Epoch {}".format(epoch))
				for batch_index in range(0, len(training_input_data), cls.config.training.batch_size):

					################ Training ###############
					x_training_batch_data = np.array(training_input_data[batch_index: batch_index+cls.config.training.batch_size])
					y_training_batch_data = np.array(training_label_data[batch_index: batch_index+cls.config.training.batch_size])

					time_str, step, loss, accuracy, precision, recall = cls.train_step(x_training_batch_data, y_training_batch_data, 
									 								cnn, session, train_op, global_step, train_summary_op, train_summary_writer)

					current_step = tf.train.global_step(session, global_step)

				if current_step % cls.config.training.evaluate_every == 0:	
					c = 0
					precision = 0
					recall = 0

					for batch_index in range(0, len(test_input_data), cls.config.training.batch_size):					
						c+=1
						x_test_batch_data = np.array(test_input_data[batch_index:cls.config.training.batch_size + batch_index])
						y_test_batch_data = np.array(test_label_data[batch_index:cls.config.training.batch_size + batch_index])

						time_str, step, loss, accuracy, t_precision, t_recall = cls.dev_step(x_test_batch_data, y_test_batch_data, 
																				cnn, session, global_step, train_summary_op, writer=train_summary_writer)		
						
						precision +=t_precision
						recall += t_recall

					precision /=c
					recall /=c

					DurationRecorder.pr_epoch_plotter(epoch, precision, recall)
					DurationRecorder.pr_epoch_logger(epoch, precision, recall)


				if current_step % cls.config.training.checkpoint_every == 0:
					path = saver.save(session, checkpoint_prefix, global_step=current_step)
					print("Saved checkpoint model to {}\n".format(path))	