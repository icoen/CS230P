# CS-230

Autumn 2018 Project

Jorge Cordero, Eddie Sun, Zoey Zhou 
{icoen, eddiesun, cuizy} @stanford.edu

<h2>Dataset and Text Preprocessing</h2>

<code> python csvreader.py </code> 

extracts the tweets from csv of the [Kaggle](https://www.kaggle.com/kapastor/democratvsrepublicantweets/version/1) dataset and removes newline character from the middle of the tweet.

<code> python gettweets.py </code> 

parses the link to the full tweet at the end of original tweets and uses BeautifulSoup to get full tweet.  (May need to first install BeautifulSoup4)

<code> python processtweets.py </code> 

Preprocesses the full dataset by removing punctuation, and splitting hashtags at the capital letters

<code> python shuffle.py </code> 

shuffles into 80/10/10 train/dev/test sets.  Saved in */trainvaltest* folder
