""" bidirectional GRU network """

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
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, GRU, GlobalMaxPooling1D, Bidirectional
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)

#inputs the positive and negative examples
tf.flags.DEFINE_string("democrat_data_file", "./../Datasets/trainvaltest/demfulltrain.txt", "Data source for Democrat data")
tf.flags.DEFINE_string("republic_data_file", "./../Datasets/trainvaltest/repfulltrain.txt", "Data source for Republican data.")

FLAGS = tf.flags.FLAGS

#biGRU network
def biGRUnet():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = Bidirectional(GRU(64, return_sequences = True), merge_mode='concat')(layer)
    layer = GlobalMaxPooling1D()(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

x_train, y_train, vocab_processor= data_helpers2.preprocess(FLAGS.democrat_data_file, FLAGS.republic_data_file)

max_words= len(vocab_processor.vocabulary_)
max_len=x_train.shape[1]

model = biGRUnet()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

x_vtext, y_val = data_helpers2.load_data_and_labels('./../Datasets/trainvaltest/demfullval.txt', './../Datasets/trainvaltest/repfullval.txt')

x_val = np.array(list(vocab_processor.transform(x_vtext)))

save_weights='weightsbgnew'
checkpointer=ModelCheckpoint(save_weights, monitor='val_acc', verbose=1, save_best_only=True)
callbacks_list=[checkpointer]
model.fit(x_train,y_train,batch_size=64,epochs=10,
          validation_data=(x_val, y_val), verbose=2, callbacks=callbacks_list)

