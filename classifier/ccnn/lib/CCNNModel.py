import json

import sklearn
import sklearn.metrics
import numpy as np
import tensorflow as tf


from .CCNNConfig import CCNNConfig

class CCNNModel(object):

	def __init__(self):
		config = CCNNConfig()

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
		for index, cv in enumerate(CCNNConfig.model.conv_layers):
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
						pooled,
						W,
						strides= config.hyper.strides,
						padding= config.hyper.padding,
						name= "conv{}".format(index))
				
				

				# Applying non-linearity
				pooled = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu{}".format(index))

				# Applying max-pooling
				if cv[-1] is not None:
					pooled = tf.nn.max_pool(
						pooled,
						ksize=[1, 1, cv[-1], 1],
						strides=[1, 1, cv[-1], 1],
						padding="VALID",
						name="pool{}".format(index))

		# Combine all pooled features

		# self.h_pool_flat = tf.reshape(pooled, [-1, config.model.num_features_total])
		shape = pooled.get_shape().as_list()
		dim = np.prod(shape[1:])
		self.h_pool_flat = tf.reshape(pooled, [-1, dim])
		self.fc_output = self.h_pool_flat

		# Fully Connected Layers
		for index, fc in enumerate(config.model.fully_connected_layers):

			# Dropout Layer
			if fc is not config.model.fully_connected_layers[-1]:
				with tf.name_scope('dropout{}'.format(index)):
					self.drop = tf.nn.dropout(self.fc_output, self.dropout_keep_prob)

			# FC Layer
			with tf.name_scope('fc{}'.format(index)):
				W = tf.Variable(tf.truncated_normal(fc, stddev=config.hyper.stddev), name="W")				

				if fc is not config.model.fully_connected_layers[-1]:
					b = tf.Variable(tf.constant(0,1, shape=[1024]), name="b")
				else:
					b = tf.Variable(tf.constant(0,1, shape=[config.model.num_of_classes]), name="b")

				if fc is not config.model.fully_connected_layers[-1]:
					xw_plus_b = tf.nn.xw_plus_b(self.drop, W, b)
					self.fc_output = tf.nn.relu(xw_plus_b, name="fc-{}-out".format(index))
				else:			
					self.scores = tf.nn.xw_plus_b(self.fc_output, W, b, name="output")
					self.predictions = tf.argmax(self.scores, 1, name="predictions")
					
		# Calculate Loss
		with tf.name_scope("loss"):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses)
		# Accuracy 
		with tf.name_scope("accuracy"):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
