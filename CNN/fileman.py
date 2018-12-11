import csv
import time
import datetime
import numpy as np

def initializer(filename = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S'), **sets): 
	if filename[-1:] == '+':
	   filename      = filename[:-1]+'-'+datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
	filename +='.csv'
	with open(filename, 'a') as csvfile:
         writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
         writer.writerow(sets.get("order"))
	return filename

def write_row_csv(file, state, **sets):
	with open(file, 'a') as csvfile:
		filewriter = csv.DictWriter(csvfile, fieldnames=sets.get("order"), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		filewriter.writerow(state)




