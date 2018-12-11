import csv
import time
import datetime
import numpy as np

def initializer(process, model, maxAccur, dictMaxAccur): 

	filename   = "%.3f" % maxAccur + model + process +'.csv'

	with open(filename, 'a') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow([maxAccur, dictMaxAccur])

	return filename

def write_row_csv(file, key, value):

	with open(file, 'a') as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow([key, value])



