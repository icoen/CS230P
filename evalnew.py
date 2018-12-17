""" evaluate single text inputs """

import csv
import numpy as np
#from attentionnoglove import RNN
from tensorflow.contrib import learn
import data_helpers2
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, TimeDistributed,Flatten, RepeatVector, Permute, Lambda, GRU
from keras.optimizers import RMSprop
import keras
import os
import tensorflow as tf
import re
import os
import trainatt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)
#suppress warnings

vocab_processor = learn.preprocessing.VocabularyProcessor.restore('vocabdict')

max_words= len(vocab_processor.vocabulary_)
max_len=65
#loading saved tokenizations and parameters

modelm=trainatt.Attennet(max_len, max_words)
# load weights
modelm.load_weights("weightsf2")
# Compile model (required to make predictions)
modelm.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
sentence=input("\n\nInput a sentence (within quotations) \n")

while sentence!='END':
    sentence = re.sub("[(),!?\'`\":;.+@#$]", " ", sentence)
    ll=[]
    ll.append(sentence.lower())
    x_val = np.array(list(vocab_processor.transform(ll)))

    predict=modelm.predict(x_val, verbose=0)
    print(str(int(100*predict[0][0]))+'% Democrat and '+str(100-int(100*predict[0][0]))+'% Republican')
    sentence=input("\n\nInput a sentence (within quotations) \n")

