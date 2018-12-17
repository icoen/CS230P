import training
import tensorflow as tf
import numpy      as np
import os
import time
import datetime
import data_helpers
from   text_cnn           import TextCNN
from   tensorflow.contrib import learn

def main(argv=None):

	choice = int(input('1) Use Defaults  2) Manually enter Hyperparameters  '))

	if choice == 2:
		embedding_dim     = int(input('Embedding dimension?:  '))
		num_filters       = int(input('Number of filters?:  '  ))
		dropout_keep_prob = float(input('Droput?:  '   ))
		l2_reg_lambda     = float(input('L2 Lambda?:  '))

		training.sethypers(embedding_dim = embedding_dim,   l2_reg_lambda = l2_reg_lambda,   
	     		       dropout_keep_prob = dropout_keep_prob, num_filters = num_filters)

	num_epochs = int(input('Number of epochs to train for?: '))
	x_train, y_train, vocab_processor, x_dev, y_dev = training.preprocess()
	training.setparams(num_epochs = num_epochs)
	training.train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
	tf.app.run()