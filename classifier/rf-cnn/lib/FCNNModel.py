import json

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from .FCNNConfig import FCNNConfig

class FCNNModel(object):

	def __init__(self):
		config = FCNNConfig()

		# Initializing input, label, and dropout
		with tf.name_scope('Input-layer'):
			self.input_x = tf.placeholder(tf.float32, [
														None, # Batch Size
														config.model.frame_size, # Width
														config.model.sequence_length # Height	
														# Channel - Added later 
													], name="input_x")
			self.input_y = tf.placeholder(tf.float32, [None, config.model.num_of_classes], name="input_y")
			self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		# Convert input into conv2d friendly format (4D)
		pooled = tf.expand_dims(self.input_x, -1) # Adds channel dimension to the input in the last axis
		pooled_outputs = []
		
 
		# Convolutional Layers
		for index, cv in enumerate(FCNNConfig.model.conv_layers):
			with tf.name_scope("conv-layer-{}".format(index)):
				if index==0:
					filter_shape = [config.model.frame_size, cv[0], 
									1, config.model.num_of_filters]

					W = tf.Variable(tf.truncated_normal(filter_shape, 
									stddev=config.hyper.stddev), dtype="float32", name = "W")
					
					b = tf.Variable(tf.constant(0.1, shape=[config.model.num_of_filters]), name="b")

					conv = tf.nn.conv2d(
						pooled,
						W,
						strides= config.hyper.strides,
						padding= config.hyper.padding,
						name= "conv{}".format(index))

				else:
					filter_shape = [1, cv[1], 
									config.model.num_of_filters, config.model.num_of_filters]

					W = tf.Variable(tf.truncated_normal(filter_shape, 
									stddev=config.hyper.stddev), dtype="float32", name = "W")

					b = tf.Variable(tf.constant(0.1, shape=[config.model.num_of_filters]), name="b")

					conv = tf.nn.conv2d(
						h,
						W,
						strides= config.hyper.strides,
						padding= config.hyper.padding,
						name= "conv{}".format(index))

				# Applying non-linearity
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu{}".format(index))

				# Applying max-pooling
				if cv[-1] is not None:
					pooled = tf.nn.max_pool(
						h,
						ksize=[1, 1, cv[-1], 1],
						strides=[1, 1, cv[-1], 1],
						padding="VALID",
						name="pool{}".format(index))

		pooled_concats = tf.nn.dropout(pooled, self.dropout_keep_prob)

		rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=config.model.num_of_filters)
		rnn_cell = tf.contrib.rnn.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_prob)

		batch_size_T = tf.shape(self.input_x)[0]

		self._initial_state = rnn_cell.zero_state(batch_size_T, tf.float32)
		inputs = tf.squeeze(pooled, [1])
		inputs = tf.split(inputs, config.model.num_of_filters, 2)


		inputs = [tf.squeeze(input_, [2]) for input_ in inputs]
		outputs, state = rnn.static_rnn(cell=rnn_cell, inputs=inputs, initial_state=self._initial_state)
		self.output = outputs[-1]

		# Calculate the last layer
		with tf.name_scope('output'):
			W = tf.Variable(tf.truncated_normal([config.model.num_of_filters, config.model.num_of_classes], stddev=config.hyper.stddev), name="W")				
			b = tf.Variable(tf.constant(0, 1, shape=[config.model.num_of_classes]), name="b")								
			self.scores = tf.nn.xw_plus_b(self.output, W, b, name="output")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")
					
		# Calculate Loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses)
		# Accuracy 
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")