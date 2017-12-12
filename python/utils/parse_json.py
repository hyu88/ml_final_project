'''
Parse data/iphone.json
'''
import json
import os
import csv

def ret_tweets(filename,is_save=False):
    datas = []
    if not os.path.exists(filename):
        print 'file not exist!'

    with open(filename,'r') as fin:
        for line in fin.readlines():
            temp=json.loads(line)['text']
            datas.append(temp)

    if is_save:
        with open('../data/iphone.csv','w') as fout:
            for line in datas:
                fout.write(line.strip().rstrip().encode('utf8'))

    return datas

if __name__=='__main__':
    tweets=ret_tweets('../data/iphone8cs.json',is_save=True)
    for retweet in tweets:
        print retweet
    print 'done'

