""" LSTM-attention network """

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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)
#suppress warnings

#inputs the positive and negative examples
tf.flags.DEFINE_string("democrat_data_file", "./Datasets/trainvaltest/demfulltrain.txt", "Data source for Democrat data")
tf.flags.DEFINE_string("republic_data_file", "./Datasets/trainvaltest/repfulltrain.txt", "Data source for Republican data.")

FLAGS = tf.flags.FLAGS

#Attention network
def Attennet(max_len, max_words):
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

if __name__ == "__main__":
    x_train, y_train, vocab_processor= data_helpers2.preprocess(FLAGS.democrat_data_file, FLAGS.republic_data_file)

    max_words= len(vocab_processor.vocabulary_)

    max_len=x_train.shape[1]
    vocab_processor.save('vocabdictnew')

    vocab_dict = vocab_processor.vocabulary_._mapping

    model = Attennet(max_len, max_words)
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=.00366),metrics=['accuracy'])

    x_vtext, y_val = data_helpers2.load_data_and_labels('./Datasets/trainvaltest/demfullval.txt', './Datasets/trainvaltest/repfullval.txt')

    x_val = np.array(list(vocab_processor.transform(x_vtext)))

    save_weights   = 'weightsfdnew'
    checkpointer   = ModelCheckpoint(save_weights, monitor = 'val_acc', verbose = 1, save_best_only = True)
    callbacks_list = [checkpointer]
    model.fit(x_train,y_train,batch_size = 128, epochs = 5, validation_data = (x_val, y_val), verbose = 2, callbacks = callbacks_list) #train the model

