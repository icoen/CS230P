import sys
sys.path.append('../CNN')
import training
import simplernnnoglove
import tensorflow as tf
import random     as rand

testCases  = 0
num_epochs = 0

embedding_dim    = 0
num_filters      = 0
dropout_keep_pro = 0

dictAvgAccurTrain = {}
listAvgAccurTrain = []
dictAvgAccurValid = {}
listAvgAccurValid = []

def searchCNN():

	x_train, y_train, vocab_processor, x_dev, y_dev = training.preprocess()

	for iterator in range(testCases):
		embedding_dim, num_filters, dropout_keep_pro, l2_reg_lambda = randomizeCNN()
		print("Embedding_dim: "    + str(embedding_dim))
		print("Num_filters: "      + str(num_filters))
		print("Dropout_keep_pro: " + str(dropout_keep_pro))
		temp_dict       = {"Embedding_dim":embedding_dim, "Num_filters":num_filters, "Dropout_keep_pro":dropout_keep_pro, "L2_reg_lambda":l2_reg_lambda}
		training.sethypers(embedding_dim = embedding_dim,   num_filters=num_filters,   dropout_keep_pro=dropout_keep_pro,   l2_reg_lambda=l2_reg_lambda)
		training.setparams(num_epochs    = num_epochs, evaluate_every=300, checkpoint_every=300)
		avgAccurTrain, avgAccurValid     = training.train()
		dictAvgAccurTrain[avgAccurTrain] = temp_dict
		dictAvgAccurValid[avgAccurValid] = temp_dict
		listAvgAccurTrain.append(avgAccurTrain)
		listAvgAccurValid.append(avgAccurValid)

	for key in dictAvgAccurTrain:
		print("Acc Train: " + str(key) + " - " + str(dictAvgAccurTrain[key]))

	for key in dictAvgAccurValid:
		print("Acc Valid: " + str(key) + " - " + str(dictAvgAccurValid[key]))

	print("Best Hyperparameters:")
	#print("Acc Train: " + str(max(listAvgAccurTrain)) + " - " + str(dictAvgAccurTrain[max(listAvgAccurTrain)]))
	#print("Acc Valid: " + str(max(listAvgAccurValid)) + " - " + str(dictAvgAccurValid[max(listAvgAccurValid)]))

def searchRNN():

	for iterator in range(testCases):
		#embedding_dim, num_filters, dropout_keep_pro, l2_reg_lambda = randomizeCNN()
		temp_dict          = {"Embedding_dim":1, "Num_filters":1, "Dropout_keep_pro":1, "L2_reg_lambda":1}
		#simplernnnoglove.sethypers(embedding_dim    = embedding_dim,   num_filters=num_filters,   dropout_keep_pro=dropout_keep_pro,   l2_reg_lambda=l2_reg_lambda)
		simplernnnoglove.setparams(num_epochs = num_epochs)
		print("QUEPEDO")
		avgAccurTrain, avgAccurValid          = simplernnnoglove.train()
		print("QUEPEDO")
		dictAvgAccurTrain[avgAccurTrain]      = temp_dict
		dictAvgAccurValid[avgAccurValid]      = temp_dict
		listAvgAccurTrain.append(avgAccurTrain)
		listAvgAccurValid.append(avgAccurValid)

	for key in dictAvgAccurTrain:
		print("Acc Train: " + str(key) + " - " + str(dictAvgAccurTrain[key]))

	for key in dictAvgAccurValid:
		print("Acc Valid: " + str(key) + " - " + str(dictAvgAccurValid[key]))

	print("Best Hyperparameters:")
	#print("Acc Train: " + str(max(listAvgAccurTrain)) + " - " + str(dictAvgAccurTrain[max(listAvgAccurTrain)]))
	#print("Acc Valid: " + str(max(listAvgAccurValid)) + " - " + str(dictAvgAccurValid[max(listAvgAccurValid)]))


def randomizeCNN():
	num_filters      = rand.randint(1, 128)
	embedding_dim    = rand.randint(1, 128) 
	dropout_keep_pro = rand.random()
	l2_reg_lambda    = rand.random()
	return embedding_dim, num_filters, dropout_keep_pro, l2_reg_lambda

def randomizeRNN():
	#min_frequency    = 
	#embedding_dim    = 
	#lstm		     =
	#dense_output     =
	dropout_keep_pro = rand.random()
	#learning_rate    =
	#batch_size       =
	return dropout_keep_pro
#return min_frequency, embedding_dim, lstm, dense_output, dropout_keep_pro, learning_rate, batch_size


def main(argv=None):

	trainModel = int(input('Model to search?: 1. CNN   -  2. RNN  -  3. Quit  '))
	testCases  = int(input('Number of test cases?: '))
	num_epochs = int(input('Epochs per test case?: '))

	if trainModel   == 1:
		searchCNN()

	elif trainModel == 2:
		searchRNN()
	

if __name__ == '__main__':
	tf.app.run()