"""hyperparameter search for LSTM attention"""

import simpleattention
import numpy      as np
import random     as rand
import tensorflow as tf
import math

model = 'ATT'

embedding_dim     = 0
num_filters       = 0
dropout_keep_prob = 0

#dictAvgAccurTrain = {}
#listAvgAccurTrain = []
#dictAvgAccurValid = {}
#listAvgAccurValid = []  

dictMaxAccurTrain = {}
listMaxAccurTrain = []
dictMaxAccurValid = {}
listMaxAccurValid = [] 

def randomize():
	temp_dict = {"LSTM": 64, "Dense_output":64, "Learning_rate":10**(1.5*rand.random()-2.5),      
		"Batch_size":rand.choice([32, 64, 128]),"Embedding_dim":rand.choice([50, 100, 200]), "Dropout_keep_prob":(rand.random()*.5+.3) ,
			 "Min_freq":0}
#hyperparameters for narrow search

	return temp_dict

"""
	temp_dict = {"LSTM":rand.choice([32, 64, 128, 256]), "Dense_output":rand.choice([64, 128, 256, 512]), "Learning_rate":10**(3*rand.random()-4),
		"Batch_size":rand.choice([32, 64, 128, 256]),"Embedding_dim":rand.choice([50, 100, 200, 300]),     "Dropout_keep_prob":(rand.random()*.8+.2) ,
"""
#hyperparameters for search


def main(argv=None):

	testCases  = int(input('Number of test cases?: '))
	num_epochs = int(input('Epochs per test case?: '))

	with open('hypersearchATTresultsnew.txt','w') as fout:

		for iterator in range(testCases):

			print("Test Case " + str(iterator+1))
			temp_dict = randomize()

			for key in temp_dict:
				print(key + ": " + str(temp_dict[key]))

			simpleattention.sethypers(lstm  = temp_dict["LSTM"], dropout_keep_prob = temp_dict["Dropout_keep_prob"], embedding_dim = temp_dict["Embedding_dim"], dense_output = temp_dict["Dense_output"], min_freq = temp_dict["Min_freq"], learning_rate = temp_dict["Learning_rate"]) 
			simpleattention.setparams(num_epochs  = num_epochs, batch_size = temp_dict["Batch_size"])
 
			maxAccurTrain, maxAccurValid          = simpleattention.train() #train with attention 
			dictMaxAccurTrain[maxAccurTrain]      = temp_dict
			dictMaxAccurValid[maxAccurValid]      = temp_dict 
			listMaxAccurTrain.append(maxAccurTrain)
			listMaxAccurValid.append(maxAccurValid)
			print ('max Acc Train:', str(maxAccurTrain))
			print ('max Acc Val:', str(maxAccurValid))

			fout.write ("Acc Train: " + str(temp_dict) + " - " + str(maxAccurTrain)+'\n')
			fout.write("Acc Valid: " + str(temp_dict) + " - " + str(maxAccurValid)+'\n\n')

		print("\nTraining Set Performance:\n")
		for key in dictMaxAccurTrain:
			print("Acc Train: " + str(key) + " - " + str(dictMaxAccurTrain[key]))
		print("\nValidation Set Performance:\n")
		for key in dictMaxAccurValid:
			print("Acc Valid: " + str(key) + " - " + str(dictMaxAccurValid[key]))

		fout.write("\n\nBest Hyperparameters:\n")
		fout.write("Acc Train: " + str(max(listMaxAccurTrain)) + " - " + str(dictMaxAccurTrain[max(listMaxAccurTrain)])+'\n')
		fout.write("Acc Valid: " + str(max(listMaxAccurValid)) + " - " + str(dictMaxAccurValid[max(listMaxAccurValid)]))
		print("\nBest Hyperparameters:\n")
		print("Acc Train: " + str(max(listMaxAccurTrain)) + " - " + str(dictMaxAccurTrain[max(listMaxAccurTrain)]))
		print("Acc Valid: " + str(max(listMaxAccurValid)) + " - " + str(dictMaxAccurValid[max(listMaxAccurValid)]))

if __name__ == '__main__':
	tf.app.run()
