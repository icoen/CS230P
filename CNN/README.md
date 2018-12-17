# CS-230

Autumn 2018 Project

Jorge Cordero, Eddie Sun, Zoey Zhou 
{icoen, eddiesun, cuizy} @stanford.edu

This project classifies political affiliation from tweets using a convolutional neural network(CNN) and a long-short-term-memory network with attention (LSTM-Attn).

<h2>CNN Hyperparameter Search</h2>
To run the Hyperparameter search, run:

<code> python hypersearchCNN.py </code> 

Parameters asked: 

testCases: Number of spawns you want to explore, 
num_epochs: Number of epochs you want each test case to train for.  

<h2>CNN Training</h2>

To run the Training, run:

<code> python runTraining.py </code> 

You will be asked if you want to use the default hyperparameters or enter yours.

Parameters asked: 

embedding_dim: Embedding dimension 
num_filters: Number of filters to usedropout_keep_prob = float(input('Droput?:  '   ))
dropout_keep_prob: Dropout Probability
l2_reg_lambda: L2 Regularization Lambda
num_epochs: Number of epochs you want to train your model.  

<h2>CNN Testing</h2>

<code> python testing.py  --test_all --checkpoint_dir="./runs/CHECKPOINT/checkpoints/" </code> 
  
Replace CHECKPOINT with the ID number of your trained model on the "runs" folder.
  
Remove <code>--test_all</code> to test for specific example tweets instead of the full testing datasets, and insert the specific tweets in the <code> testing.py </code> file.

