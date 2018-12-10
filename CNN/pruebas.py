import numpy as np
import random as rand

listCost 	= []
dictAvgCost = {}
listAvgCost = []


dictAvgAccurTrain = {}
listAvgAccurTrain = []
dictAvgAccurValid = {}
listAvgAccurValid = []

def randomize():
	temp_dict = {"Embedding_dim":rand.choice([0, 1, 2]), 
				   "Num_filters":rand.choice([0, 1, 2])}
	return temp_dict

def main(argv=None):

	testCases  = int(input('Number of test cases?: '))
	num_epochs = int(input('Epochs per test case?: '))

	for iterator in range(testCases):
		temp_dict = randomize()
		if not (temp_dict in dictAvgAccurTrain.values()):
			for acc, params in temp_dict.items():
				print(acc + ": " + str(params))

			avgAccurTrain, avgAccurValid     = rand.random(), rand.random()
			dictAvgAccurTrain[avgAccurTrain] = temp_dict
			dictAvgAccurValid[avgAccurValid] = temp_dict
			listAvgAccurTrain.append(avgAccurTrain)
			listAvgAccurValid.append(avgAccurValid)

	print("\nTraining Set Performance:\n")
	for key in dictAvgAccurTrain:
		print("Acc Train: " + str(key) + " - " + str(dictAvgAccurTrain[key]))

	print("\nValidation Set Performance:\n")
	for key in dictAvgAccurValid:
		print("Acc Valid: " + str(key) + " - " + str(dictAvgAccurValid[key]))

	print("\nBest Hyperparameters:\n")
	print("Acc Train: " + str(max(listAvgAccurTrain)) + " - " + str(dictAvgAccurTrain[max(listAvgAccurTrain)]))
	print("Acc Valid: " + str(max(listAvgAccurValid)) + " - " + str(dictAvgAccurValid[max(listAvgAccurValid)]))
	print("\n")

	#for values in dictAvgAccurTrain:
	#	print("Acc Train: " + str(key) + " - " + str(dictAvgAccurTrain[key]))

main()






#for values in dictAvgAccurTrain:
#		print("Acc Train: " + str(key) + " - " + str(dictAvgAccurTrain[key]))




	










# for cont1 in range(100):
# 	embedding_dim, num_filters, dropout_keep_pro, l2_reg_lambda = run.randomize()
# 	for cont2 in range(1000):
# 		cost=rand.random()
# 		listCost.append(cost)
# 		#print("Cost: "     + str(cost))
# 	avgCost=np.sum(listCost)/(len(listCost))
# 	#print("Embedding_dim: " + str(embedding_dim) + "  -  Num_filters: "      + str(num_filters)      + 
#     #	                	                       "  -  Dropout_keep_pro: " + str(dropout_keep_pro) +  
#     #	                	                       "  -  L2_reg_lambda: "    + str(l2_reg_lambda)    +  
#      #   	                	                   "  -  Average Cost: "     + str(avgCost))

# 	temp_dict = {"Embedding_dim":embedding_dim, "Num_filters":num_filters, "Dropout_keep_pro":dropout_keep_pro, "L2_reg_lambda":l2_reg_lambda}
# 	dictAvgCost[avgCost] = temp_dict
# 	listAvgCost.append(avgCost)


# for key in dictAvgCost:
# 	print(str(key) + " - " + str(dictAvgCost[key]))

# print("Best Hyperparameter:")
# print(str(max(listAvgCost)) + " - " + str(dictAvgCost[max(listAvgCost)]))