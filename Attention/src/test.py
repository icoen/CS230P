import pandas as pd
import regex as re
import numpy as np
import pickle
import keras.preprocessing as kpp
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding, Conv1D
from keras.layers import MaxPooling1D, Flatten, RepeatVector, Permute, Input
from keras.layers import TimeDistributed, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.activations import softmax, tanh
from sklearn.model_selection import train_test_split


def main():
    
    # filepaths
    extracted_tweets_filepath = '../data/ExtractedTweets.csv'
    processed_tweets_savepath = '../data/TweetsProcessed.h5'
    glove_filepath= '../data/glove.6B.50d.txt'
    glove_model_savepath = '../data/glove.twitter.27B.50d.h5'
    save_weights = '../data/weights.h5'
    
    # bools
    load_preprocess_tweets_from_scratch = False
    load_glove_from_scratch = False
    load_weights_bool = False
    embedding_train_bool = True
    
    # hyperparameters
    max_words_in_sentence = 40
    vocab_size_max = 10000
    lr = 0.001 # learning rate
    beta_1 = 0.9 # adam optimizer
    beta_2 = 0.999 # adam optimizer
    valid_split = 0.1 # % of train data to set aside for validation
    test_split = 0.1 # % of all data to set aside for test
    num_epochs = 20
    batch_size = 128   
    
    # load tweets and preprocess tweets
    if load_preprocess_tweets_from_scratch == True:
        rep_tweets, dem_tweets = load_tweets(extracted_tweets_filepath)
        rep_tweets, dem_tweets = preprocess_tweets(rep_tweets, dem_tweets)
        save_data(processed_tweets_savepath, [rep_tweets, dem_tweets])
    else:
        rep_tweets, dem_tweets = open_data(processed_tweets_savepath)
    
    # make labels
    rep_labels = np.zeros((len(rep_tweets), ))
    dem_labels = np.ones((len(dem_tweets), ))
    y = np.concatenate((rep_labels, dem_labels))
    
    # load glove
    if load_glove_from_scratch == True:
        embeddings_index = loadGloveModel(glove_filepath) 
        save_data(glove_model_savepath, embeddings_index)
    else:
        embeddings_index = open_data(glove_model_savepath)
        print (len(embeddings_index)," words loaded!")
    glove_dim = embeddings_index['hello'].shape[0]
        
    # tokenize with keras
    vocab_size, embedding_matrix, padded_docs = tokenize(rep_tweets, dem_tweets, vocab_size_max, embeddings_index, max_words_in_sentence, glove_dim)
    x = padded_docs
    
    # split data    
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = test_split, random_state = 1)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size = valid_split, random_state = 1)
    
    # build_RNN
#    model = build_RNN(vocab_size, glove_dim, embedding_matrix, max_words_in_sentence, embedding_train_bool)
    model = build_attn_RNN(vocab_size, glove_dim, embedding_matrix, max_words_in_sentence, embedding_train_bool)
    
    # train/valid RNN
    model, history = train_valid_RNN(model, X_train, Y_train, lr, beta_1, beta_2, num_epochs, 
                                         batch_size, [X_valid, Y_valid], save_weights, load_weights_bool)
    
#    # predict
#    predictions = model.predict(X_valid, batch_size)
#    predictions = predictions[predictions <= 0.5]
#    incorrect_preds = np.nonzero(predictions != Y_valid)
#    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))X_valid[incorrect_preds]
    

    
    
    
        
        
        
def build_RNN(vocab_size, glove_dim, embedding_matrix, max_words_in_sentence, embedding_train_bool):
    print('\nBuilding model...')
    
    tf.reset_default_graph()
    keras.backend.clear_session()

    model = Sequential()
    model.add(Embedding(vocab_size, glove_dim, input_length = max_words_in_sentence, 
                        weights = [embedding_matrix], trainable = embedding_train_bool))
    
    model.add(LSTM(64, return_sequences = False, activation = myAct)) 
    
    model.add(Dense(256, name = 'FC1'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, name = 'out_layer'))
    model.add(Activation('sigmoid'))

    model.summary()
    
    return model

def build_attn_RNN(vocab_size, glove_dim, embedding_matrix, max_words_in_sentence, embedding_train_bool):
    print('\nBuilding attention model...')
    
    inputs = Input(shape = [max_words_in_sentence], dtype='int32')
    
    # embedded = Embedding(vocab_size, glove_dim, input_length = max_words_in_sentence, weights = [embedding_matrix], trainable = embedding_train_bool)(inputs)
    embedded = Embedding(vocab_size, glove_dim, input_length = max_words_in_sentence, trainable = embedding_train_bool)(inputs)
    
#    activations = Conv1D(64, 5, activation = 'relu')(embedded)
#    activations = MaxPooling1D(pool_size = 4)(activations)
    
    activations = LSTM(64, return_sequences = True)(embedded)
    
    attention = TimeDistributed(Dense(1, activation='tanh'))(activations) 
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(64)(attention)
    attention = Permute([2, 1])(attention)

    merged = keras.layers.Multiply()([activations, attention])
    merged = Lambda(lambda xin: keras.backend.sum(xin, axis=-2), output_shape=(64,))(merged)
    
    probabilities = Dense(256, name = 'FC1', activation = 'relu')(merged)
    probabilities = Dropout(0.5)(probabilities)
    
    probabilities = Dense(1, name = 'out_layer', activation = 'sigmoid')(probabilities)
    
    model = Model(inputs = inputs, outputs = probabilities)
    
    model.summary()
    
    return model

def train_valid_RNN(model, x_train, y_train, lr, beta_1, beta_2, num_epochs, 
                    batch_size, valid_data, save_weights, load_weights_bool):
    
    checkpointer = ModelCheckpoint(save_weights, monitor = 'val_loss', verbose = 1, save_best_only = True)
    print('Training Model...')
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr, beta_1, beta_2), metrics = ['accuracy'])
    
    # load saved weights as starting point
    if load_weights_bool == True:
        print ('Loading saved weights as initial weights.')
        model.load_weights(save_weights)
    
    history = model.fit(x_train, y_train, batch_size = batch_size, epochs = num_epochs, 
                        validation_data = valid_data, callbacks = [checkpointer])
    
    print('Model training complete.', '\n')
    
    return model, history   




def myAct(out):
    return softmax(tanh(out))
        
def tokenize(rep_tweets, dem_tweets, vocab_size_max, embeddings_index, max_words_in_sentence, glove_dim):      
    
    docs = rep_tweets + dem_tweets
    
    t = kpp.text.Tokenizer(num_words = vocab_size_max)
    t.fit_on_texts(docs)    
    encoded_docs = t.texts_to_sequences(docs)
    padded_docs = kpp.sequence.pad_sequences(encoded_docs, maxlen = max_words_in_sentence, padding = 'post')
    vocab_size = len(t.word_index) + 1

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, glove_dim))
    for word, i in t.word_index.items():
    	embedding_vector = embeddings_index.get(word)
    	if embedding_vector is not None:
    		embedding_matrix[i] = embedding_vector
    
    print ('Vocab size:', vocab_size)
    print ('Embedding matrix shape:', embedding_matrix.shape)
    
    return vocab_size, embedding_matrix, padded_docs
                


def load_tweets(filepath):
    
    print ('\nLoading tweets...')

    rep_tweets = []
    dem_tweets = []
    
    data = pd.read_csv(filepath).values
    
    for i in range(data.shape[0]):
        if data[i, 0] == 'Republican':
            rep_tweets.append(data[i, 2])
        if data[i, 0] == 'Democrat':
            dem_tweets.append(data[i, 2])
            
    print ('# of Republican tweets loaded for training:', len(rep_tweets))
    print ('# of Democratic tweets loaded for training:', len(dem_tweets))
    
    return rep_tweets, dem_tweets

def preprocess_tweets(rep_tweets, dem_tweets):
    
    # source: https://gist.github.com/tokestermw/cb87a97113da12acb388
    
    def hashtag(text):
        text = text.group()
        hashtag_body = text[1:]
        if hashtag_body.isupper():
            result = " {} ".format(hashtag_body.lower())
        else:
            result = " ".join(["<hashtag>"] + re.split(r"(?=[A-Z])", hashtag_body, flags=FLAGS))
        return result
    
    def allcaps(text):
        text = text.group()
        return text.lower() + " <allcaps>"
    
    def tokenize(text):
        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`\-]?"
    
        # function so code less repetitive
        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=FLAGS)
    
        text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
        text = re_sub(r"@\w+", "<user>")
        text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
        text = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
        text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
        text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
        text = re_sub(r"/"," / ")
        text = re_sub(r"<3","<heart>")
        text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
        text = re_sub(r"#\S+", hashtag)
        text = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
        text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
        text = re_sub(r"([A-Z]){2,}", allcaps)
    
        return text.lower()
    
    print ('\nPreprocessing tweets...')
    
    FLAGS = re.MULTILINE | re.DOTALL
    for i in range(len(rep_tweets)):
        rep_tweets[i] = tokenize(rep_tweets[i])
    for i in range(len(dem_tweets)):
        dem_tweets[i] = tokenize(dem_tweets[i])
    
    print ('Tweets preprocessed.')
    
    return rep_tweets, dem_tweets
    
def loadGloveModel(gloveFile):
    print ("\nLoading Glove vectors")
    
    embeddings_index = dict()
    f = open(gloveFile, encoding = 'utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index)) 
    
    return embeddings_index
    
def save_data(save_filename, save_obj):
    print('\nSaving data...')
    
    with open(save_filename, 'wb') as fi:
        pickle.dump(save_obj, fi)
        
    print('Save data complete. Object saved as:', save_filename)

def open_data(save_filename):
    print('\nOpening', save_filename, '...')
    
    with open(save_filename, 'rb') as fi:
        load_temp = pickle.load(fi)
    loaded = load_temp      

    print('Open data complete.')
    
    return loaded

    
if __name__ == '__main__':
    main()
