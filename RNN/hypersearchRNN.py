import simplernnnoglove
import tensorflow as tf
import numpy      as np
import random     as rand

embedding_dim     = 0
num_filters       = 0
dropout_keep_prob = 0

dictAvgAccurTrain = {}
listAvgAccurTrain = []
dictAvgAccurValid = {}
listAvgAccurValid = []

def randomize():
	temp_dict = {"Min_freq":rand.randint(1, 3)               ,     "Embedding_dim":rand.choice([ 50, 100, 200, 300])            ,       "LSTM":rand.choice([ 32,  64, 128, 256]), 
	         "Dense_output":rand.choice([ 64, 128, 256, 512]), "Dropout_keep_prob":rand.random() , "Learning_rate":rand.random(), "Batch_size":rand.choice([ 32,  64, 128, 256])}
	return temp_dict

def main(argv=None):

	testCases  = int(input('Number of test cases?: '))
	num_epochs = int(input('Epochs per test case?: '))

	#x_train, y_train, vocab_processor, x_dev, y_dev = training.preprocess()

	for iterator in range(testCases):
		temp_dict = randomize()

		for key in temp_dict:
			print(key + ": " + str(temp_dict[key]))

		simplernnnoglove.sethypers(min_freq   = temp_dict["Min_freq"],          embedding_dim = temp_dict["Embedding_dim"],              lstm = temp_dict["LSTM"], 
			                     dense_output = temp_dict["Dense_output"], dropout_keep_prob  = temp_dict["Dropout_keep_prob"], learning_rate = temp_dict["Learning_rate"])
		simplernnnoglove.setparams(batch_size = temp_dict["Batch_size"],           num_epochs = num_epochs)
		avgAccurTrain, avgAccurValid          = simplernnnoglove.train()
		dictAvgAccurTrain[avgAccurTrain]      = temp_dict
		dictAvgAccurValid[avgAccurValid]      = temp_dict 
		listAvgAccurTrain.append(avgAccurTrain)
		listAvgAccurValid.append(avgAccurValid)

	for key in dictAvgAccurTrain:
		print("Acc Train: " + str(key) + " - " + str(dictAvgAccurTrain[key]))

	for key in dictAvgAccurValid:
		print("Acc Valid: " + str(key) + " - " + str(dictAvgAccurValid[key]))

	print("Best Hyperparameters:")
	print("Acc Train: " + str(max(listAvgAccurTrain)) + " - " + str(dictAvgAccurTrain[max(listAvgAccurTrain)]))
	print("Acc Valid: " + str(max(listAvgAccurValid)) + " - " + str(dictAvgAccurValid[max(listAvgAccurValid)]))

if __name__ == '__main__':
	tf.app.run()