import training
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
	temp_dict = {"Embedding_dim":rand.choice([50, 100, 200, 300]), "Dropout_keep_prob":rand.random(), 
	               "Num_filters":rand.choice([ 16, 32,  64, 128]),     "L2_reg_lambda":rand.random()}
	                           
	return temp_dict

def main(argv=None):

	testCases  = int(input('Number of test cases?: '))
	num_epochs = int(input('Epochs per test case?: '))

	x_train, y_train, vocab_processor, x_dev, y_dev = training.preprocess()

	for iterator in range(testCases):
		temp_dict = randomize()

		for key in temp_dict:
			print(key + ": " + str(temp_dict[key]))
		
		training.sethypers(embedding_dim = temp_dict["Embedding_dim"],            num_filters = temp_dict["Num_filters"],  
					   dropout_keep_prob = temp_dict["Dropout_keep_prob"],      l2_reg_lambda = temp_dict["L2_reg_lambda"])
		training.setparams(num_epochs    = num_epochs, evaluate_every = 300, checkpoint_every = 300)
		avgAccurTrain, avgAccurValid     = training.train(x_train, y_train, vocab_processor, x_dev, y_dev)
		dictAvgAccurTrain[avgAccurTrain] = temp_dict
		dictAvgAccurValid[avgAccurValid] = temp_dict
		listAvgAccurTrain.append(avgAccurTrain)
		listAvgAccurValid.append(avgAccurValid)

	print("Training Set Performance:/n")
	for key in dictAvgAccurTrain:
		print("Acc Train: " + str(key) + " - " + str(dictAvgAccurTrain[key]))

	print("/nValidation Set Performance:")
	for key in dictAvgAccurValid:
		print("Acc Valid: " + str(key) + " - " + str(dictAvgAccurValid[key]))

	print("/nBest Hyperparameters:/n")
	print("Acc Train: " + str(max(listAvgAccurTrain)) + " - " + str(dictAvgAccurTrain[max(listAvgAccurTrain)]))
	print("Acc Valid: " + str(max(listAvgAccurValid)) + " - " + str(dictAvgAccurValid[max(listAvgAccurValid)]))

if __name__ == '__main__':
	tf.app.run()