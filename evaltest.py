""" evaluates test set """

import csv
import numpy as np
#from attentionnoglove import RNN
from tensorflow.contrib import learn
import data_helpers2
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, TimeDistributed,Flatten, RepeatVector, Permute, Lambda, GRU
from keras.optimizers import RMSprop
import keras
import trainatt

vocab_processor = learn.preprocessing.VocabularyProcessor.restore('vocabdict')
x_vtext, y_val = data_helpers2.load_data_and_labels('./Datasets/trainvaltest/demfulltest.txt', './Datasets/trainvaltest/repfulltest.txt')

x_val = np.array(list(vocab_processor.transform(x_vtext)))
max_words= len(vocab_processor.vocabulary_)

max_len=x_val.shape[1]

modelm=trainatt.Attennet(max_len, max_words)
# load weights from last run
modelm.load_weights("weightsf2")
# Compile model (required to make predictions)
modelm.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

evald=modelm.predict(x_val, verbose=1)

demcor=0
repcor=0
for i in range(len(y_val)):
    if (evald[i]>=0.5 and y_val[i]==1):
        demcor+=1
    elif (evald[i]<0.5 and y_val[i]==0):
            repcor+=1

print ('accuracy is: ' + str(float((demcor+repcor))/len(y_val)))
print ('correct predictions for democrats: ' + str(demcor))
print ('correct predictions for republicans: '+ str(repcor))
