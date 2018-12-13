import tensorflow as tf
import os
import time
import datetime
import data_helpers2
from tensorflow.contrib import learn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, TimeDistributed,Flatten, RepeatVector, Permute, Lambda, GRU
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping,  ModelCheckpoint
import keras

#inputs the positive and negative examples
tf.flags.DEFINE_string("positive_data_file", "./../Datasets/trainvaltest/demfulltrain.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./../Datasets/trainvaltest/repfulltrain.txt", "Data source for the negative data.")

FLAGS = tf.flags.FLAGS
def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers2.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_train = x[shuffle_indices]
    y_train = y[shuffle_indices]

    del x, y

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    return x_train, y_train, vocab_processor

#RNN network
def RNN():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,300,input_length=max_len)(inputs)
    activations = LSTM(64, return_sequences = True)(layer)

    attention = TimeDistributed(Dense(1, activation='tanh'))(activations)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(64)(attention)
    attention = Permute([2, 1])(attention)
 
    merged = keras.layers.Multiply()([activations, attention])
    merged = Lambda(lambda xin: keras.backend.sum(xin, axis=-2), output_shape=(64,))(merged)
    probabilities = Dense(64, name = 'FC1', activation = 'relu')(merged)
    probabilities = Dropout(0.44)(probabilities)
 
    probabilities = Dense(1, name = 'out_layer', activation = 'sigmoid')(probabilities)

    model = Model(inputs=inputs,outputs=probabilities)
    return model

x_train, y_train, vocab_processor= preprocess()
max_words= len(vocab_processor.vocabulary_)
max_len=x_train.shape[1]
#x_train= np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#print x_train.shape

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=.00366),metrics=['accuracy'])

x_vtext, y_val = data_helpers2.load_data_and_labels('./../Datasets/trainvaltest/demfullval.txt', './../Datasets/trainvaltest/repfullval.txt')
x_ttext, y_test = data_helpers2.load_data_and_labels('./../Datasets/trainvaltest/demfulltest.txt', './../Datasets/trainvaltest/repfulltest.txt')

x_val = np.array(list(vocab_processor.transform(x_vtext)))
x_test= np.array(list(vocab_processor.transform(x_ttext)))

save_weights   = 'weightsf2'
checkpointer   = ModelCheckpoint(save_weights, monitor = 'val_acc', verbose = 1, save_best_only = True)
callbacks_list = [checkpointer]
model.fit(x_train,y_train,batch_size = 128,     epochs = 5,
							   validation_data = (x_val, y_val),         verbose = 2, callbacks = callbacks_list) #train the model

#model.fit(x_train,y_train,batch_size=128,epochs=10,
#          validation_data=(x_val, y_val), verbose=2)



modelm=RNN()
# load weights
modelm.load_weights("weightsf2")
# Compile model (required to make predictions)
modelm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

predictions=modelm.predict(x_val)

predictionst=modelm.predict(x_test)

print (predictions[0])

def printerr(fname,predictions, y_val, x_val):
    fpred=open(fname,'w')

    j=0
    x_errors=[]

    for i in range(len(predictions)):
       # if (predictions[i]>=0.5 and y_val[i]==0) or (predictions[i]<0.5 and y_val[i]==1):
	#    x_errors.append(str(y_val[i])+', '+x_val[i])
	 #   j+=1
        x_errors.append(str(predictions[i])+', '+str(y_val[i])+', '+x_val[i])
	#np.savetxt('predictions.csv',x_errors)
    for errors in x_errors:
	fpred.write(errors)
	fpred.write('\n')
    fpred.close()
#,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
printerr('predictval.txt', predictions, y_val, x_vtext)
printerr('predicttest.txt', predictionst, y_test, x_ttext)

