# Predicting Political Affiliation from Tweets

CS-230 Autumn 2018 Project

Jorge Cordero, Eddie Sun, Zoey Zhou 
{icoen, eddiesun, cuizy} @stanford.edu

<h2>Task</h2>

This project classifies political affiliation from tweets using the long-short-term-memory network with attention (LSTM-Attn).  A convolutional neural network(CNN) model is in the CNN folder.  (The previous two are implemented with hyperparameter search)  Additional models such as LSTM, bi-directional LSTM, GRU, bi-directional GRU are in the RNN folder.  All code is implemented in Keras and Tensorflow.  Can be run in python 2 or python 3

<h2>Dataset</h2>

We use a balanced dataset comprised of 86,460 labeled tweets from [Kaggle](https://www.kaggle.com/kapastor/democratvsrepublicantweets/version/1). After text preprocessing, the dataset is divided in 6 files, democrat and republican files for each of the train development and test sets with an 80/10/10 split.  These are in the Datasets/trainvaltest folder.

<h2>LSTM-Attention Hyperparameter Search</h2>
To run the Hyperparameter search, run:

<code> python hypersearchATT.py </code> 

Parameters asked: 

testCases: Number of spawns you want to explore. <br>
num_epochs: Number of epochs you want each test case to train for.  

<h2>LSTM-Attn. Training</h2>

To run the Training, run:

<code> python trainatt.py </code> 

using the best hyperparameters found during the hyperparameter search as documented in the report for this project.  Saves best weight as *weightsfdnew*.

<h2>LSTM-Attn. Testing</h2>

<h4>Merge the weight file</h4>

For both types of testing you must first run 

```python mergeweights.py```

which will combine *weights1* and *weights2* into one file *weightsf2* which can then be loaded as weights.  We had to split the weight files due to size limitations of github.

<h4>Run pre-loaded model on test set</h4>

<code> python evaltest.py  </code> 
  
loads the tokenization from *vocabdict* and weights from *weightsf2*, the weights we use in our report.  And evaluates on the test data.  Outputs the accuracy, and the number of correct predictions for democrats and republicans.  You can change *vocabdict*, *weightsf2* and the test data files within the python file to evaluate user specified test data on newly trained models.

<h4>Run pre-loaded model on user input</h4>

<code> python evalnew.py  </code> 

loads the tokenization from *vocabdict* and weights from *weightsf2*.  Repeatedly asks for a sentence to predict in terminal and outputs the p value as a percentage for democrat and republican.  Can input any sentences surrounded by quotes.  Enter 'END' to end.  

Sample as follows
```
input: "We have an affordable housing crisis"
output: 89% Democrat and 11% Republican 
```
