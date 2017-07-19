import datetime
import os 

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from .CSVManipulator import CSVManipulator

class DurationRecorder:

	prev_plot_time = None
	prev_log_time = None
	diff_list = []
	step_list = []
	loss_list = []
	prec_list = []
	recall_list = []
	acc_list = []

	@classmethod
	def start_log(cls):
		cls.prev_plot_time = datetime.datetime.now()
		cls.prev_log_time = cls.prev_plot_time

	@classmethod
	def al_epoch_logger(cls, step, acc, loss, log_dir="./training.log", csv_dir="./training_log.csv"):	
		"""
			Accuracy & Loss logger
		"""
		curr_time = datetime.datetime.now()
		if cls.prev_log_time is None:
			diff = 0
			cls.prev_log_time = curr_time
		else:
			diff = curr_time - cls.prev_log_time
			diff = diff.total_seconds()

			hour,reminder = divmod(diff, 3600)
			minute, second = divmod(reminder, 60)	

		if log_dir is not None:
			log_info = "Step {}\n".format(step)
			log_info += "loss: {} - accuracy: {}\n".format(loss, acc)
			log_info += "interval: {} hour(s) : {} minute(s) : {} second(s)\n\n".format(hour, minute, second)

			writer = open(log_dir, 'a')
			writer.write(log_info)
			writer.close()		

		if csv_dir is not None:
			data = np.array([step, diff, loss, acc, hour, minute, second])
			with open(csv_dir, 'a') as output:
				csv_writer = csv.writer(output, delimiter=",")
				csv_writer.writerow(data)

		cls.prev_log_time = curr_time

	@classmethod
	def al_epoch_plotter(cls, step, acc, loss):
		"""
			Accuracy & Loss plotter
		"""
		curr_time = datetime.datetime.now()
		if cls.prev_plot_time is None:
			diff = 0
			cls.prev_plot_time = curr_time
		else:
			diff = curr_time - cls.prev_plot_time
			diff = diff.total_seconds()

		cls.step_list.append(step)
		cls.diff_list.append(diff)
		cls.loss_list.append(loss)
		cls.acc_list.append(acc)

		loss_plot = plt.subplot(2, 2, 1)
		plt.title("Loss Function")
		plt.xlabel("Step")
		plt.ylabel("Loss")

		avg_loss = sum(cls.loss_list) / float(len(cls.loss_list))
		loss_plot.add_artist(AnchoredText("Avg Loss: {}".format(round(loss, 5)), loc=3))
		plt.plot(cls.step_list, cls.loss_list, c="red")

		accuracy_plot = plt.subplot(2, 2, 2)
		plt.title("Accuracy")
		plt.xlabel("Step")
		plt.ylabel("Accuracy")

		avg_acc = sum(cls.acc_list) / float(len(cls.acc_list))
		accuracy_plot.add_artist(AnchoredText("Avg Accuracy: {}".format(round(accuracy, 5)), loc=3))
		plt.plot(cls.step_list, cls.acc_list, c="blue")

		interval_plot = plt.subplot(2, 2, 3)
		plt.title("Interval")
		plt.xlabel("Step")
		plt.ylabel("Seconds")

		avg_diff = sum(cls.diff_list) / float(len(cls.diff_list))
		interval_plot.add_artist(AnchoredText("Avg Interval: {} seconds".format(round(avg_diff, 2)), loc=3))
		plt.plot(cls.step_list, cls.diff_list, c="green")

		plt.draw()
		plt.pause(0.00001)

		cls.prev_plot_time = curr_time

	@classmethod
	def pr_epoch_logger(cls, step, precision, recall, log_dir="./training.log", csv_dir="./training_log.csv"):	
		"""
			Precision & Recall Logger
		"""	
		curr_time = datetime.datetime.now()
		if cls.prev_log_time is None:
			diff = 0
			cls.prev_log_time = curr_time
		else:
			diff = curr_time - cls.prev_log_time
			diff = diff.total_seconds()

			hour,reminder = divmod(diff, 3600)
			minute, second = divmod(reminder, 60)	

		if log_dir is not None:
			log_info = "Step {}\n".format(step)
			log_info += "precision: {} - recall: {}\n".format(precision, recall)
			log_info += "interval: {} hour(s) : {} minute(s) : {} second(s)\n\n".format(hour, minute, second)

			print(log_info)

			writer = open(log_dir, 'a')
			writer.write(log_info)
			writer.close()		

		if csv_dir is not None:
			data = np.array([step, diff, precision, recall, hour, minute, second])

			with open(csv_dir, 'a') as output:
				csv_writer = csv.writer(output, delimiter=",")
				csv_writer.writerow(data)

		cls.prev_log_time = curr_time

	@classmethod
	def pr_epoch_plotter(cls, step, precision, recall, log_dir="./training.log", csv_dir="./training_log.csv"):
		"""
			Precision & Recall Plotter
		"""	
		curr_time = datetime.datetime.now()
		if cls.prev_plot_time is None:
			diff = 0
			cls.prev_plot_time = curr_time
		else:
			diff = curr_time - cls.prev_plot_time
			diff = diff.total_seconds()

		cls.step_list.append(step)
		cls.diff_list.append(diff)
		cls.prec_list.append(precision)
		cls.recall_list.append(recall)

		precision_plot = plt.subplot(2, 2, 1)
		plt.title("Precision")
		plt.xlabel("Step")
		plt.ylabel("Precision")

		avg_prec = sum(cls.prec_list) / float(len(cls.prec_list))
		precision_plot.add_artist(AnchoredText("Avg Precision: {}".format(round(avg_prec, 5)), loc=3))
		plt.plot(cls.step_list, cls.prec_list, c="red")

		recall_plot = plt.subplot(2, 2, 2)
		plt.title("Recall")
		plt.xlabel("Step")
		plt.ylabel("Recall")

		avg_recall = sum(cls.recall_list) / float(len(cls.recall_list))
		recall_plot.add_artist(AnchoredText("Avg Recall: {}".format(round(avg_recall, 5)), loc=3))
		plt.plot(cls.step_list, cls.recall_list, c="blue")

		interval_plot = plt.subplot(2, 2, 3)
		plt.title("Interval")
		plt.xlabel("Step")
		plt.ylabel("Seconds")

		avg_diff = sum(cls.diff_list) / float(len(cls.diff_list))
		interval_plot.add_artist(AnchoredText("Avg Interval: {} seconds".format(round(avg_diff, 2)), loc=3))
		plt.plot(cls.step_list, cls.diff_list, c="green")

		plt.draw()
		plt.pause(0.00001)

		cls.prev_plot_time = curr_time

	@classmethod
	def pr_csv_plotter(cls, np_data, step_index=0, interval_index=1, precision_index=2, recall_index=3, every_step=1):
		print("Creating Plot")
		cls.step_list = np_data[::every_step, step_index]
		cls.diff_list = np_data[::every_step, interval_index]
		cls.prec_list = np_data[::every_step, precision_index]
		cls.recall_list = np_data[::every_step, recall_index]

		print(len(cls.step_list))
		print(len(cls.diff_list))
		print(len(cls.prec_list))
		print(len(cls.recall_list))

		precision_plot = plt.subplot(2, 2, 1)
		plt.title("Precision")
		plt.xlabel("Step")
		plt.ylabel("Precision")

		avg_prec = sum(cls.prec_list) / float(len(cls.prec_list))
		precision_plot.add_artist(AnchoredText("Avg Precision: {}".format(round(avg_prec, 5)), loc=3))
		plt.plot(cls.step_list, cls.prec_list, c="red")

		recall_plot = plt.subplot(2, 2, 2)
		plt.title("Recall")
		plt.xlabel("Step")
		plt.ylabel("Recall")

		avg_recall = sum(cls.recall_list) / float(len(cls.recall_list))
		recall_plot.add_artist(AnchoredText("Avg Recall: {}".format(round(avg_recall, 5)), loc=3))
		plt.plot(cls.step_list, cls.recall_list, c="blue")

		interval_plot = plt.subplot(2, 2, 3)
		plt.title("Interval")
		plt.xlabel("Step")
		plt.ylabel("Seconds")

		avg_diff = sum(cls.diff_list) / float(len(cls.diff_list))
		interval_plot.add_artist(AnchoredText("Avg Interval: {} seconds".format(round(avg_diff, 2)), loc=3))
		plt.plot(cls.step_list, cls.diff_list, c="green")

		plt.draw()
		plt.pause(0.00001)

