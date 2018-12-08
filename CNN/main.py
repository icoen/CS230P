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
	x_train, y_train, vocab_processor, x_dev, y_dev = training.preprocess()
	training.train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
	tf.app.run()