'''
Main file of the project
'''

import pickle
import os
from sklearn.metrics import accuracy_score
# load file for lstm, svm, random forest
from utils import pre_nltk_data

# load data for baselines
from utils.parse_json import ret_tweets
filename='data/iphone.json'
tweets=ret_tweets(filename,is_save=False)
keyword='iphone8'
time='lastweek'


# train
# algorithm = 'baseline'
# algorithm = 'naivebayes'
# algorithm = 'LSTM'
# algorithm = 'SVM'
# algorithm = 'RForest'
# algorithm = 'GaussBayes'
#algorithm = 'All'

# algorithm needs to be specified
algorithm = 'LSTM'
is_lstm=True

# determine whether to valid or test
filename='data/processed_full_training_dataset.csv'
#filename='data/processed_pre_iphone.csv'
testPath='data/test_text.csv'
is_test=True
is_valid=True

if (algorithm == 'baseline' ):
    from classifier import baseline_classifier
    if not os.path.exists(filename):
        print 'file not exist!'
    bc = baseline_classifier.BaselineClassifier(tweets, keyword, time,filename)

    # save as json
    if is_test:
        bc.classify(is_file=True)
    bc.accuracy()

if (algorithm == 'naivebayes' ):
    from classifier import naive_bayes_classifier
    if not os.path.exists(filename):
        print 'training file empty!'

    # trainingDataFile = 'data/full_training_dataset.csv'
    classifierDumpFile = 'data/test/naivebayes_test_model.pickle'
    trainingRequired = 1
    nb = naive_bayes_classifier.NaiveBayesClassifier(tweets, keyword, time, \
                                                     filename, classifierDumpFile, trainingRequired)
    # save as json
    if is_test:
        nb.classify()
    nb.accuracy()

else:

    # load data
    print 'loading data...'
    if is_lstm:
        trainX, trainY, dictionary, num_labels,testX = pre_nltk_data.stem_token_data(filename,testPath,is_lstm)
    else:
        trainX, trainY,testX = pre_nltk_data.stem_token_data(filename, testPath,is_lstm)

    # generate valid data
    if is_valid:
        from utils import pre_nltk_data
        trainX,validX,trainY,validY=pre_nltk_data.train_valid_split(trainX,trainY)


    if (algorithm=='LSTM' ):
        # num_epochs=1
        # is_lstm=True
        # from classifier import lstm_classifier
        # clf=lstm_classifier.LSTM_classifier(trainX,trainY,dictionary,num_labels,num_epochs)
        import matplotlib.pyplot as plt
        num_epochs = 3
        #epoch=[1]
        #epoch = [1,2,3,4,5,6,7,8]
        loss = []
        test_acc = []
        val_acc = []
        layer=8
        layers=[8]
        #layers=[2,4,6,8,10,12]
        #for num_epochs in epoch:
        for layer in layers:
            is_lstm = True
            from classifier import lstm_classifier

            clf = lstm_classifier.LSTM_classifier(trainX, trainY, dictionary, num_labels, num_epochs,layer)
            loss_and_metrics = clf.evaluate(validX, validY)
            loss.append(loss_and_metrics[0])
            val_acc.append(loss_and_metrics[1])
            print 'loss: {}, valid accuracy with LSTM: {}'.format(loss_and_metrics[0], loss_and_metrics[1])
            if is_test:
                from utils.pre_nltk_data import store_result

                print 'testing ...'
                pred = clf.predict(testX)
                if algorithm != 'ALL':
                    pre_acc = store_result(tweets, pred, algorithm, is_lstm=is_lstm)
                    test_acc.append(pre_acc)

        plt.plot(layers, loss, linewidth=2.0, color='green')
        #plt.plot(layers, test_acc, linewidth=2.0, color='blue')
        plt.plot(layers, val_acc, linewidth=2.0, color='red')
        plt.legend(['loss', 'test_acc', 'val_acc'], loc='upper right')
        plt.xlabel('layer')
        plt.show()
    elif(algorithm=='SVM' or algorithm=='All'):
        from classifier import svm_classifier

        clf=svm_classifier.SVM_classifier(trainX,trainY)


    elif (algorithm=='RForest' or algorithm=='All'):
        from classifier import random_forest_classifier

        max_depth=2
        clf=random_forest_classifier.RF_classifier(trainX,trainY,max_depth)


    elif (algorithm=='GaussBayes' or algorithm=='All'):

        from classifier import gauss_bayes
        clf=gauss_bayes.Gauss_bayes_classifier(trainX,trainY)









