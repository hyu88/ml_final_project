import pickle
import re
import sys

import classifier_helper
import csv
import json
reload(sys)
sys.setdefaultencoding = 'utf-8'

#start class
class BaselineClassifier:
    """ Classifier using baseline method """
    #variables    
    #start __init__
    def __init__(self, data, keyword, time,trainingDataFile):
        #Instantiate classifier helper        
        self.helper = classifier_helper.ClassifierHelper('data/feature_list.txt')
        #Remove duplicates

        self.lenTweets = len(data)
        self.origTweets = self.getUniqData(data)
        self.tweets = self.getProcessedTweets(self.origTweets)
        
        self.results = {}
        self.neut_count = [0] * self.lenTweets
        self.pos_count = [0] * self.lenTweets
        self.neg_count = [0] * self.lenTweets

        self.time = time
        self.trainingDataFile = trainingDataFile
        self.keyword = keyword

    #end

    def getUniqData(self, data):
        uniq_data = {}
        i = 0
        # each d is a sentence
        for d in data:
            u = []
            for element in d.split(' '):
                if element not in u:
                    u.append(element)
            # end inner loop
            uniq_data[i] = u
            i += 1
        # end outer loop

        return uniq_data
        # end

        # start getProcessedTweets

    def getProcessedTweets(self, data):

        tweets = {}
        count = 0
        # each i is an index of a sentence
        for i in data:
            tw = []
            for t in data[i]:
                tw.append(self.helper.process_tweet(t))
            tweets[count] = tw
            count += 1
        # end loop
        return tweets


    
    #start classify
    def classify(self,is_file=False,data=None):
        #load positive keywords file          
        inpfile = open("data/pos_mod.txt", "r")            
        line = inpfile.readline()
        positive_words = []
        while line:
            positive_words.append(line.strip())
            line = inpfile.readline()
            
        #load negative keywords file    
        inpfile = open("data/neg_mod.txt", "r")            
        line = inpfile.readline()
        negative_words = []
        while line:
            negative_words.append(line.strip())
            line = inpfile.readline()
        #start processing each tweet

        if is_file==False:
            neg_words = [word for word in negative_words if (self.string_found(word, data))]
            pos_words = [word for word in positive_words if (self.string_found(word, data))]
            if (len(pos_words) > len(neg_words)):
                label = 'positive'

            elif (len(pos_words) < len(neg_words)):
                label = 'negative'
            else:
                if (len(pos_words) > 0 and len(neg_words) > 0):
                    label = 'positive'
                else:
                    label = 'neutral'
            return label

        elif is_file==True:
            for i in self.tweets:
                tw = self.tweets[i]
                tw=' '.join(tw)
                neg_words = [word for word in negative_words if(self.string_found(word, tw))]
                pos_words = [word for word in positive_words if(self.string_found(word, tw))]
                if(len(pos_words) > len(neg_words)):
                    label = 'positive'
                    self.pos_count[i] += 1
                elif(len(pos_words) < len(neg_words)):
                    label = 'negative'
                    self.neg_count[i] += 1
                else:
                    if(len(pos_words) > 0 and len(neg_words) > 0):
                        label = 'positive'
                        self.pos_count[i] += 1
                    else:
                        label = 'neutral'
                        self.neut_count[i] += 1
                result = {'text': tw, 'label': label}
                self.results[i] = result

        with open('data/test/result_baseline.json', 'w') as fout:
            json.dump(self.results,fout)
        return self.results


    def getMinCount(self, trainingDataFile):
        fp = open( trainingDataFile, 'rb' )
        reader = csv.reader( fp, delimiter=',', quotechar='"', escapechar='\\' )
        neg_count, pos_count, neut_count = 0, 0, 0
        for row in reader:
            sentiment = row[0]
            if(sentiment == 'neutral'):
                neut_count += 1
            elif(sentiment == 'positive'):
                pos_count += 1
            elif(sentiment == 'negative'):
                neg_count += 1
        #end loop
        return min(neg_count, pos_count, neut_count)


    # start getFilteredTrainingData
    def getFilteredTrainingData(self, trainingDataFile):
        fp = open(trainingDataFile, 'rb')
        min_count = self.getMinCount(trainingDataFile)
        min_count = 40000
        neg_count, pos_count, neut_count = 0, 0, 0

        reader = csv.reader(fp, delimiter=',', quotechar='"', escapechar='\\')
        tweetItems = []
        count = 1
        for row in reader:
            processed_tweet = self.helper.process_tweet(row[1])
            sentiment = row[0]

            if (sentiment == 'neutral'):
                if (neut_count == int(min_count)):
                    continue
                neut_count += 1
            elif (sentiment == 'positive'):
                if (pos_count == min_count):
                    continue
                pos_count += 1
            elif (sentiment == 'negative'):
                if (neg_count == min_count):
                    continue
                neg_count += 1

            tweet_item = processed_tweet, sentiment
            tweetItems.append(tweet_item)
            count += 1
        # end loop
        return tweetItems
        # end



    def accuracy(self):
        tweets = self.getFilteredTrainingData(self.trainingDataFile)
        total = 0
        correct = 0
        wrong = 0
        self.accuracy = 0.0
        for (t, l) in tweets:
            label = self.classify(is_file=False,data=t)
            if(label == l):
                correct+= 1
            else:
                wrong+= 1
            total += 1
        #end loop
        self.accuracy = (float(correct)/total)*100
        print 'Total = %d, Correct = %d, Wrong = %d, Accuracy = %.2f' % \
                                                (total, correct, wrong, self.accuracy)

    #start substring whole word match
    def string_found(self, string1, string2):
        if re.search(r"\b" + re.escape(string1) + r"\b", string2):
            return True
        return False
    #end
    
    #start writeOutput
    def writeOutput(self, filename, writeOption='w'):
        fp = open(filename, writeOption)
        for i in self.results:
            res = self.results[i]
            for j in res:
                item = res[j]
                text = item['text'].strip()
                label = item['label']
                writeStr = text+" | "+label+"\n"
                fp.write(writeStr)
            #end inner loop
        #end outer loop      
    #end writeOutput

#end class    
