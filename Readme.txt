
In this project file, it mainly runs for LSTM experiments.
###################################### 
The main file: analyze2.py
######################################
**********Parameters******************
filename: the training data path
testPath: the test data path
is_test=True: after the training process, the procedure will execute the testing part. is_valid=True: during the training process, the procedure will execute the validation part.
num_epochs: the epoch value that the classifier to run
epoch: the set of different epoch values, which is used to test the training results with different epoch values. 
layer: the LSTM network layer number, which can be set to different values to adjust the training effects. 
###################################### 
The network building file: classifier/lstm_classifier.py
######################################
**********Parameters******************
To train a Bidirectional LSTM classifier, use
model.add(Bidirectional(LSTM(layer, return_sequences=True)))
model.add(Bidirectional(LSTM(layer)))

To train a Unidirectional LSTM classifier, use
model.add(LSTM(layer, dropout_W=0.2, dropout_U=0.2))

###################################### 
The network building file: utils/pre_nltk_data.py
######################################
**********Parameters******************
Function stem_token_data preprocess training data and test data to tokens and stems to facilitate the training process, the outputs are the inputs of lstm classifiers.

Function store_result stores the predicted result into a file.


###################################### 
Another main file: analyze.py
######################################
This file is used to train the LSTM neural network with GloVe embedding technique.

In order to run this file, you need to change the name of pre_nltk_data3.py to pre_nltk_data.py, change the name of lstm_classifier3.py to lstm_classifier.py.

##################################################
Training data: processed_full_training_dataset.csv
Test data: processed_pre_iphone.csv 
##################################################

###################################### 
Raw tweets acquiring : Crawler.py
######################################
Change the json file into csv file: parse_json.py
######################################
Preprocess tweets: Peprocess_Tweets.py
######################################