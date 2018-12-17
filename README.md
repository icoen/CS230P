# Predicting Political Affiliation from Tweets

CS-230 Autumn 2018 Project

Jorge Cordero, Eddie Sun, Zoey Zhou 
{icoen, eddiesun, cuizy} @stanford.edu

<h2>Task</h2>

This project classifies political affiliation from tweets using the long-short-term-memory network with attention (LSTM-Attn).  A convolutional neural network(CNN) model is in the CNN folder.  (The previous two are implemented with hyperparameter search)  Additional models such as LSTM, bi-directional LSTM, GRU, bi-directional GRU are in the RNN folder.

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

using the best hyperparameters found during the hyperparameter search as documented in the report for this project.

<h2>LSTM-Attn. Testing</h2>

<code> python evaltest.py  </code> 
  
loads the tokenization from `vocabdict' and weights from `weightsf2', the weights we use in our report.  And evaluates on the test data.  Outputs the accuracy, and the number of correct predictions for democrats and republicans.  You can change vocabdict, weightsf2 and the test data files within the python file to evaluate using new trained models and on different test data.
  
<code> python evalnew.py  </code> 

loads the tokenization from `vocabdict' and weights from `weightsf2'.  Repeatedly asks for a sentence to predict in terminal and outputs the p value for democrat and republican.  Can input any sentences surrounded by quotes.  Enter `END' to end.  
