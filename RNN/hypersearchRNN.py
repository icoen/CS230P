import simplernnnoglove
import numpy      as np
import random     as rand
import fileman	  as fm
import tensorflow as tf

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
	temp_dict = {"LSTM":rand.choice([32, 64, 128, 256]), "Dense_output":rand.choice([64, 128, 256, 512]), "Learning_rate":10**(-4*rand.random()),      
		   "Batch_size":rand.choice([32, 64, 128, 256]),"Embedding_dim":rand.choice([50, 100, 200, 300]),     "Dropout_keep_prob":rand.random() ,
			 "Min_freq":rand.randint(1, 3)}
	return temp_dict

def main(argv=None):

	testCases  = int(input('Number of test cases?: '))
	num_epochs = int(input('Epochs per test case?: '))

	#x_train, y_train, vocab_processor, x_dev, y_dev = training.preprocess()

	for iterator in range(testCases):

		print("Test Case " + str(iterator+1))
		temp_dict = randomize()

		for key in temp_dict:
			print(key + ": " + str(temp_dict[key]))

		simplernnnoglove.sethypers(lstm = temp_dict["LSTM"], dropout_keep_prob = temp_dict["Dropout_keep_prob"], embedding_dim = temp_dict["Embedding_dim"],                   
						   dense_output = temp_dict["Dense_output"],  min_freq = temp_dict["Min_freq"],          learning_rate = temp_dict["Learning_rate"]) 
		simplernnnoglove.setparams(num_epochs = num_epochs,         batch_size = temp_dict["Batch_size"])
		#avgAccurTrain, avgAccurValid          = simplernnnoglove.train()
		#dictAvgAccurTrain[avgAccurTrain]      = temp_dict
		#dictAvgAccurValid[avgAccurValid]      = temp_dict 
		#listAvgAccurTrain.append(avgAccurTrain)
		#listAvgAccurValid.append(avgAccurValid)

		maxAccurTrain, maxAccurValid          = simplernnnoglove.train()
		dictMaxAccurTrain[maxAccurTrain]      = temp_dict
		dictMaxAccurValid[maxAccurValid]      = temp_dict 
		listMaxAccurTrain.append(maxAccurTrain)
		listMaxAccurValid.append(maxAccurValid)

	#for key in dictAvgAccurTrain:
	#	print("Acc Train: " + str(key) + " - " + str(dictAvgAccurTrain[key]))

	#for key in dictAvgAccurValid:
	#	print("Acc Valid: " + str(key) + " - " + str(dictAvgAccurValid[key]))

	#print("Best Hyperparameters:")
	#print("Acc Train: " + str(max(listAvgAccurTrain)) + " - " + str(dictAvgAccurTrain[max(listAvgAccurTrain)]))
	#print("Acc Valid: " + str(max(listAvgAccurValid)) + " - " + str(dictAvgAccurValid[max(listAvgAccurValid)]))

	print("\nTraining Set Performance:\n")
	for key in dictMaxAccurTrain:
		print("Acc Train: " + str(key) + " - " + str(dictMaxAccurTrain[key]))

	print("\nValidation Set Performance:\n")
	for key in dictMaxAccurValid:
		print("Acc Valid: " + str(key) + " - " + str(dictMaxAccurValid[key]))

	print("\nBest Hyperparameters:\n")
	print("Acc Train: " + str(max(listMaxAccurTrain)) + " - " + str(dictMaxAccurTrain[max(listMaxAccurTrain)]))
	print("Acc Valid: " + str(max(listMaxAccurValid)) + " - " + str(dictMaxAccurValid[max(listMaxAccurValid)]))

if __name__ == '__main__':
	tf.app.run()