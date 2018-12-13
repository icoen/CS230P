import training
import numpy      as np
import random     as rand
import fileman	  as fm
import tensorflow as tf

model = 'CNN'

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
	#temp_dict = {"Embedding_dim":rand.choice([50, 100, 200, 300]), "Dropout_keep_prob":rand.random(), 
	#			   "Num_filters":rand.choice([ 16, 32,  64, 128]),     "L2_reg_lambda":rand.random()}

	temp_dict = {"Embedding_dim":100, "Dropout_keep_prob":.519, 
				   "Num_filters":128,     "L2_reg_lambda":.345}
	
	#temp_dict = {"Embedding_dim":100, "Dropout_keep_prob":rand.uniform(0.3, 0.7), 
	#			   "Num_filters":128,     "L2_reg_lambda":rand.uniform(0.1, 0.5)}
							   
	return temp_dict

def main(argv=None):

	testCases  = int(input('Number of test cases?: '))
	num_epochs = int(input('Epochs per test case?: '))

	x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test = training.preprocess()

	for iterator in range(testCases):

		print("Test Case " + str(iterator+1))
		temp_dict = randomize()

		for key in temp_dict:
			print(key + ": " + str(temp_dict[key]))
		
		training.sethypers(embedding_dim = temp_dict["Embedding_dim"],            num_filters = temp_dict["Num_filters"],  
					   dropout_keep_prob = temp_dict["Dropout_keep_prob"],      l2_reg_lambda = temp_dict["L2_reg_lambda"])
		training.setparams(num_epochs    = num_epochs, evaluate_every = 300, checkpoint_every = 300)
		#avgAccurTrain, avgAccurValid     = training.train(x_train, y_train, vocab_processor, x_dev, y_dev)
		#dictAvgAccurTrain[avgAccurTrain] = temp_dict
		#dictAvgAccurValid[avgAccurValid] = temp_dict
		#listAvgAccurTrain.append(avgAccurTrain)
		#listAvgAccurValid.append(avgAccurValid)

		#maxAccurTrain, maxAccurValid = rand.random(), rand.random()
		maxAccurTrain, maxAccurValid, maxAccurTest    = training.train(x_train, y_train, vocab_processor, x_dev, y_dev, x_test, y_test)
		dictMaxAccurTrain[maxAccurTrain] = temp_dict
		dictMaxAccurValid[maxAccurValid] = temp_dict
		listMaxAccurTrain.append(maxAccurTrain)
		listMaxAccurValid.append(maxAccurValid)
		print (str(maxAccurTrain)+'\t'+ str(maxAccurValid)+'\t'+ str(maxAccurTest))

	print("\nBest Hyperparameters:\n")
	maxAccurTrains = max(listMaxAccurTrain)
	maxAccurValids = max(listMaxAccurValid)
	print("Acc Train: " + str(maxAccurTrains) + " - " + str(dictMaxAccurTrain[maxAccurTrains]))
	print("Acc Valid: " + str(maxAccurValids) + " - " + str(dictMaxAccurValid[maxAccurValids]))

	print("\nTraining Set Performances:\n")
	
	filename=fm.initializer(model = model, process = "Train", maxAccur = maxAccurTrains, dictMaxAccur = dictMaxAccurTrain[maxAccurTrains])

	for key, value in dictMaxAccurTrain.items():
		print("Acc Train: " + str(key) + " - " + str(value))
		fm.write_row_csv(filename, key = key, value = value)

	print("\nValidation Set Performances:\n")

	filename=fm.initializer(model = model, process = "Valid", maxAccur = maxAccurValids, dictMaxAccur = dictMaxAccurTrain[maxAccurTrains])

	for key, value in dictMaxAccurValid.items():
		print("Acc Valid: " + str(key) + " - " + str(value))
		fm.write_row_csv(filename, key = key, value = value)

if __name__ == '__main__':
	tf.app.run()
