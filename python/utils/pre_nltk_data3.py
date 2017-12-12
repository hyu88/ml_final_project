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
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
import pandas as pd
import os

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 1045
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def stem_token_data(filePath,testPath,is_lstm=False):
    # import libraries
    from keras.utils import np_utils
    print('Indexing word vectors.')

    embeddings_index = {}
    f = open(os.path.join('glove.6B/', 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    print('Processing training dataset')
    t=Tokenizer(num_words=MAX_NB_WORDS)
    # ------------------------------------------------------------------------
    train_df = pd.read_csv(filePath, sep=',', names=['Sentiment', 'Phrase'])
    raw_docs_train = train_df['Phrase'].values
    df_test = pd.read_csv(testPath, names=['Phrase'])
    docs=train_df['Phrase'].values
    raw_test = df_test['Phrase'].values

    train_df['Sentiment'] = train_df['Sentiment'].map({'positive': 0, 'negative': 1, 'neutral': 2})
    labels = train_df['Sentiment'].values
    sentiment_train = train_df['Sentiment'].values

    t.fit_on_texts(docs)
    vocab_size=len(t.word_index)+1
    encoded_docs=t.texts_to_sequences(docs)
    word_index = t.word_index
    print('Found %s unique tokens.' % len(word_index))
    print (encoded_docs)

    data = pad_sequences(encoded_docs, maxlen=MAX_SEQUENCE_LENGTH)

    labels = to_categorical(np.asarray(labels))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    t.fit_on_texts(raw_test)
    vocab_size_test = len(t.word_index) + 1
    encoded_docs_test = t.texts_to_sequences(raw_test)
    word_index_test = t.word_index
    print('Found %s unique tokens.' % len(word_index_test))
    print (encoded_docs_test)

    data_test = pad_sequences(encoded_docs_test, maxlen=MAX_SEQUENCE_LENGTH)
    testX=data_test

    X_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    X_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]
    num_labels = len(np.unique(sentiment_train))
    print 'num_labels: {}'.format(num_labels)
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    print num_words
    print t.word_index.items().__len__()

    for word, i in t.word_index.items():
        if i >= num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    print X_train.shape
    print y_train.shape
    print 'embedding matrix shape: {}'.format(embedding_matrix.shape)
    if is_lstm:
        return X_train, y_train,X_val, y_val, testX,num_labels,num_words,embedding_matrix



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





