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
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
from rnn_utils import *


#inputs the positive and negative examples
tf.flags.DEFINE_string("positive_data_file", "./twtdemtrain3.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./twtreptrain3.txt", "Data source for the negative data.")
#tf.flags.DEFINE_string("positive_data_file", "./rt-polarity.pos", "Data source for the positive data.")
#tf.flags.DEFINE_string("negative_data_file", "./rt-polarity.neg", "Data source for the negative data.")



FLAGS = tf.flags.FLAGS


#use Glove embeddings
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./../../glove.twitter.27B/glove.twitter.27B.50d.txt')

def sentences_to_indices(X, word_to_index, max_len):
  
    #m = X.shape[0]                                   # number of training examples
    m= len(X)

    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape ( 1 line)
    X_indices = np.zeros((m, max_len))
    
    for i in range(m):                               # loop over training examples
        
        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        tempX=X[i].encode('ascii', 'ignore')
        sentence_words =tempX.lower().split()
        
        #print (sentence_words)
        #print (sentence_words[0])
        # Initialize j to 0
        j = 0
        
        # Loop over the words of sentence_words
        for w in sentence_words:
            if w in word_to_index:
            # Set the (i,j)th entry of X_indices to the index of the correct word.

                X_indices[i, j] = word_to_index[w]
            # Increment j to j + 1
                j = j+1
            
    ### END CODE HERE ###
    
    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    
    #Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.
    
    #Arguments:
    #word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    #word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    #Returns:
    #embedding_layer -- pretrained layer Keras instance
        
    vocab_len = len(word_to_index) + 1                  # adding 1 to fit Keras embedding (requirement)
    emb_dim = word_to_vec_map["cucumber"].shape[0]      # define dimensionality of your GloVe word vectors (= 50)
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len, emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable. Use Embedding(...). Make sure to set trainable=False. 
    embedding_layer = Embedding( vocab_len,emb_dim ,trainable=False)
    
    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))
    
    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer

#RNN network
def RNN():
    #inputs = Input(name='inputs',shape=[max_len])
    sentence_indices = Input(shape=[max_len], dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    #layer = Embedding(max_words,50,input_length=max_len)(inputs) 
    layer = LSTM(64)(embeddings)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=sentence_indices,outputs=layer)
    return model


print("Loading data...")
x_text, y = data_helpers2.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
# most words/sentence
max_len = max([len(x.split(" ")) for x in x_text])

#x_train, y_train, vocab_processor= preprocess()
#max_words= len(vocab_processor.vocabulary_)
#max_len=x_train.shape[1]
#print (max_len)
#print (x_text[1])
#print (x_text[1:3])
#x_train= np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

#print x_train.shape

model = RNN()
model.summary()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.01),metrics=['accuracy'])

x_train = sentences_to_indices(x_text, word_to_index, max_len)
#Y_train_oh = convert_to_one_hot(Y_train, C = 5)
print(x_train[1:2])
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_train = x_train[shuffle_indices]
y_train = y[shuffle_indices]

model.fit(x_train,y_train,batch_size=64,epochs=20,
          validation_split=0.1) #train the model

#,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
