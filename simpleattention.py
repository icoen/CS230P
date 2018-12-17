""" LSTM-Attention hyperparameter training"""

import os
import time
import datetime
import data_helpers2
import numpy             as np
import seaborn           as sns
import tensorflow        as tf
import matplotlib.pyplot as plt
import keras
from   tensorflow.contrib      import learn
from   keras.models            import Model
from   keras.callbacks         import History
from   keras.optimizers        import RMSprop
from   sklearn.model_selection import train_test_split
from   keras.callbacks         import EarlyStopping, ModelCheckpoint
from   keras.layers            import LSTM, Activation, Dense, Dropout, Input, Embedding, TimeDistributed, Flatten, RepeatVector, Permute, Lambda 

#inputs the democrat and republican examples
tf.flags.DEFINE_string("democrat_data_file", "./Datasets/trainvaltest/demfulltrain.txt", "Data source for the democrat data.")
tf.flags.DEFINE_string("republic_data_file", "./Datasets/trainvaltest/repfulltrain.txt", "Data source for the republic data.")

# Model Hyperparameters
tf.flags.DEFINE_float("min_freq"         ,   2, "Minimum words to be valid (default: 2)")
tf.flags.DEFINE_float("embedding_dim"    ,  50, "Dimensionality of character embedding (default: 50)")
tf.flags.DEFINE_float("lstm"             ,  64, "LSTM size (default: 64)")
tf.flags.DEFINE_float("dense_output"     , 256, "Dense Output (default: 256)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate"    , 0.001, "Dropout keep probability (default: 0.5)")

# Training parameters
tf.flags.DEFINE_integer("batch_sizeRNN", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochsRNN",  1, "Number of training epochs (default: 1)")

FLAGS = tf.flags.FLAGS
history = History()

def sethypers(min_freq = 2, embedding_dim = 50, lstm = 64, dense_output = 256, dropout_keep_prob = 0.5, learning_rate = 0.001):
	 FLAGS.min_freq          = min_freq
	 FLAGS.embedding_dim     = embedding_dim
	 FLAGS.lstm              = lstm
	 FLAGS.dense_output      = dense_output
	 FLAGS.dropout_keep_prob = dropout_keep_prob
	 FLAGS.learning_rate     = learning_rate

def setparams(batch_size = 64, num_epochs = 1):
	 FLAGS.batch_sizeRNN    = batch_size
	 FLAGS.num_epochsRNN    = num_epochs

def train():
	#Attention network
	def Attennet():
		inputs = Input(name = 'inputs', shape = [max_len])
		layer  = Embedding(max_words, FLAGS.embedding_dim, input_length = max_len)(inputs)
		activations  = LSTM(FLAGS.lstm, return_sequences = True)(layer)
		attention = TimeDistributed(Dense(1, activation='tanh'))(activations) #attention layer
		attention = Flatten()(attention)
		attention = Activation('softmax')(attention)
		attention = RepeatVector(FLAGS.lstm)(attention)
		attention = Permute([2, 1])(attention)
		merged = keras.layers.Multiply()([activations, attention])
		merged = Lambda(lambda xin: keras.backend.sum(xin, axis=-2), output_shape=(FLAGS.lstm,))(merged)
		layer  = Dense(FLAGS.dense_output, name = 'FC1')(merged) #dense layer
		layer  = Activation('relu')(layer)
		layer  = Dropout(FLAGS.dropout_keep_prob)(layer) #dropout
		layer  = Dense(1, name = 'out_layer')(layer)
		layer  = Activation('sigmoid')(layer) #predictions
		model  = Model(inputs = inputs,outputs = layer)
		return model

	x_train, y_train, vocab_processor= data_helpers2.preprocess(FLAGS.democrat_data_file, FLAGS.republic_data_file, FLAGS.min_freq)

	max_words = len(vocab_processor.vocabulary_)
	max_len   = x_train.shape[1]

	x_vtext, y_val = data_helpers2.load_data_and_labels('./Datasets/trainvaltest/demfullval.txt', './Datasets/trainvaltest/repfullval.txt') #load dev data

	x_val = np.array(list(vocab_processor.transform(x_vtext))) #tokenize with training data tokenizer

	model = Attennet()
	model.summary()
	model.compile(loss = 'binary_crossentropy',  optimizer = RMSprop(lr=FLAGS.learning_rate),  metrics = ['accuracy'])

	save_weights   = 'weightslstmatt'  #save best weight using validation accuracy
	checkpointer   = ModelCheckpoint(save_weights, monitor = 'val_acc', verbose = 1, save_best_only = True)
	callbacks_list = [checkpointer]
	mod = model.fit(x_train,y_train,batch_size = FLAGS.batch_sizeRNN,     epochs = FLAGS.num_epochsRNN,
							   validation_data = (x_val, y_val),         verbose = 0, callbacks = callbacks_list) #train the model
	

	listAccurTrain = mod.history['acc']
	listAccurValid = mod.history['val_acc']

	#Calculating Maximum Accuracies for current model
	print("LIST")
	print(listAccurTrain)
	print(listAccurValid)
	print("LIST")

	maxAccurTrain = max(listAccurTrain)
	maxAccurValid = max(listAccurValid)

	return maxAccurTrain, maxAccurValid

