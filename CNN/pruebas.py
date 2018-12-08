import numpy as np
import train
import tensorflow as tf
import random as rand
import pandas as pd
import run

listCost 	= []
dictAvgCost = {}
listAvgCost = []


for cont1 in range(100):
	embedding_dim, num_filters, dropout_keep_pro, l2_reg_lambda = run.randomize()
	for cont2 in range(1000):
		cost=rand.random()
		listCost.append(cost)
		#print("Cost: "     + str(cost))
	avgCost=np.sum(listCost)/(len(listCost))
	#print("Embedding_dim: " + str(embedding_dim) + "  -  Num_filters: "      + str(num_filters)      + 
    #	                	                       "  -  Dropout_keep_pro: " + str(dropout_keep_pro) +  
    #	                	                       "  -  L2_reg_lambda: "    + str(l2_reg_lambda)    +  
     #   	                	                   "  -  Average Cost: "     + str(avgCost))

	temp_dict = {"Embedding_dim":embedding_dim, "Num_filters":num_filters, "Dropout_keep_pro":dropout_keep_pro, "L2_reg_lambda":l2_reg_lambda}
	dictAvgCost[avgCost] = temp_dict
	listAvgCost.append(avgCost)


for key in dictAvgCost:
	print(str(key) + " - " + str(dictAvgCost[key]))

print("Best Hyperparameter:")
print(str(max(listAvgCost)) + " - " + str(dictAvgCost[max(listAvgCost)]))