# pre-process data using third-party library
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from gensim import corpora
from keras.preprocessing import sequence
import pandas as pd

def stem_token_data(filePath,testPath,is_lstm=False):
    # import libraries
    from keras.utils import np_utils

    # ------------------------------------------------------------------------
    train_df = pd.read_csv(filePath, sep=',', names=['Sentiment', 'Phrase'])
    df_test = pd.read_csv(testPath, names=['Phrase'])
    raw_docs_train = train_df['Phrase'].values
    raw_test = df_test['Phrase'].values

    train_df['Sentiment'] = train_df['Sentiment'].map({'positive': 0, 'negative': 1, 'neutral': 2})
    sentiment_train = train_df['Sentiment'].values


    num_labels = len(np.unique(sentiment_train))
    print 'num_labels: {}'.format(num_labels)

    # text pre-processing
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
    stemmer = SnowballStemmer('english')
    #nltk.stem.snowballStemmer deal with different kinds of languages
    print "pre-processing train docs..."
    processed_docs_train = []
    for doc in raw_docs_train:
        try:
            doc = unicode(doc, errors='replace')
            tokens = word_tokenize(doc)
            filtered = [word for word in tokens if word not in stop_words]
            stemmed = [stemmer.stem(word) for word in filtered]
            processed_docs_train.append(stemmed)
        except:
            print "Can't encode error"

    print "pre-processing test docs..."
    processed_docs_test = []
    for doc in raw_test:
        try:
            doc = unicode(doc, errors='replace')
            tokens = word_tokenize(doc)
            filtered = [word for word in tokens if word not in stop_words]
            stemmed = [stemmer.stem(word) for word in filtered]
            processed_docs_test.append(stemmed)
        except:
            print "Can't encode error test"

    processed_docs_all = np.concatenate((processed_docs_train, processed_docs_test), axis=0)
    print 'num of docs: %d' % len(processed_docs_all)
    dictionary = corpora.Dictionary(processed_docs_all)

    dictionary_size = len(dictionary.keys())
    print "dictionary size: ", dictionary_size
    # dictionary.save('dictionary.dict')
    # corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    print "converting to token ids..."
    word_id_train, word_id_len = [], []
    for doc in processed_docs_train:
        word_ids = [dictionary.token2id[word] for word in doc]

        word_id_train.append(word_ids)
        word_id_len.append(len(word_ids))

    word_id_test, word_ids = [], []
    for doc in processed_docs_test:
        word_ids = [dictionary.token2id[word] for word in doc]
        word_id_test.append(word_ids)
        word_id_len.append(len(word_ids))

    seq_len = np.round((np.mean(word_id_len) + 2 * np.std(word_id_len))).astype(int)

    # pad sequences
    word_id_train = sequence.pad_sequences(np.array(word_id_train), maxlen=seq_len)
    word_id_test = sequence.pad_sequences(np.array(word_id_test), maxlen=seq_len)

    y_train_enc = np_utils.to_categorical(sentiment_train, num_labels)
    print type(y_train_enc)
    print word_id_train.shape
    print y_train_enc.shape

    if is_lstm:
        return word_id_train, y_train_enc, dictionary_size, num_labels,word_id_test
    else:
        return word_id_train,sentiment_train,word_id_test



'''
train valid data generation
'''
def train_valid_split(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid= train_test_split(X, y,
                                     test_size = 0.30, random_state = 42,shuffle=True)
    return X_train,X_valid,y_train,y_valid

# 'positive': 0, 'negative': 1, 'neutral': 2
def store_result(docs,pred,saveFile,is_lstm):
    import os
    saveFile=os.path.join('data/test/',saveFile+'_pred_result.txt')
    correct=0
    pos = 0
    neg = 0
    neu = 0
    fout=open(saveFile,'w')
    y_test = pd.read_csv("data/test_label.csv",names=["label"])
    ys=y_test["label"].values
    for doc,pre,y in zip(docs,pred,ys):
        y=int(y)
        if is_lstm:
            pre=np.argmax(pre)
        doc=doc.encode('utf-8')

        # if pre==0:
        #     fout.write('positive' + ',' + doc)
        # elif pre==1:
        #     fout.write('negative' + ',' + doc)
        # elif pre==2:
        #     fout.write('neutral' + ',' + doc)

        if pre == 0:
            fout.write('0')
            if pre==y:
                pos+=1
        elif pre == 1:
            fout.write('1')
            if pre==y:
                neg+=1
        elif pre == 2:
            fout.write('2')
            if pre==y:
                neu+=1
        if pre==y:
           correct+=1
        fout.write('\n')
    precision=correct/(1.0*len(ys))
    precision_neg=neg/(1.0*47)
    precision_pos=pos/(1.0*91)
    precision_neu=neu/(1.0*63)
    print 'test_precision is %f'%precision
    print 'test_precision_pos is %f' % precision_pos
    print 'test_precision_neu is %f' % precision_neu
    print 'test_precision_neg is %f' % precision_neg
    fout.close()

    print 'predictions saved!'
    return precision





