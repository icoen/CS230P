import training
import tensorflow as tf
import numpy      as np
import random     as rand

embedding_dim    = 0
num_filters      = 0
dropout_keep_pro = 0

dictAvgAccurTrain = {}
listAvgAccurTrain = []
dictAvgAccurValid = {}
listAvgAccurValid = []

def randomize():
	#def sethyper(embedding_dim=128, filter_sizes="3,4,5", num_filters=128, dropout_keep_pro=0.5, l2_reg_lambda=0.5):
	embedding_dim    = rand.randint(1, 128) #filter_sizes=
	num_filters      = rand.randint(1, 128)
	dropout_keep_pro = rand.random()
	l2_reg_lambda    = rand.random()
	return embedding_dim, num_filters, dropout_keep_pro, l2_reg_lambda


def main(argv=None):

	testCases  = int(input('Number of test cases?: '))
	num_epochs = int(input('Epochs per test case?: '))

	x_train, y_train, vocab_processor, x_dev, y_dev = training.preprocess()

	for iterator in range(testCases):
		embedding_dim, num_filters, dropout_keep_pro, l2_reg_lambda = randomize()
		print("Embedding_dim: "    + str(embedding_dim))
		print("Num_filters: "      + str(num_filters))
		print("Dropout_keep_pro: " + str(dropout_keep_pro))
		temp_dict          = {"Embedding_dim":embedding_dim, "Num_filters":num_filters, "Dropout_keep_pro":dropout_keep_pro, "L2_reg_lambda":l2_reg_lambda}
		training.sethyper(embedding_dim     = embedding_dim,   num_filters=num_filters,   dropout_keep_pro=dropout_keep_pro,   l2_reg_lambda=l2_reg_lambda)
		training.setparams(num_epochs       = num_epochs, evaluate_every=300, checkpoint_every=300)
		avgAccurTrain, avgAccurValid        = training.train(x_train, y_train, vocab_processor, x_dev, y_dev)
		dictAvgAccurTrain[avgAccurTrain]    = temp_dict
		dictAvgAccurValid[avgAccurValid]    = temp_dict
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