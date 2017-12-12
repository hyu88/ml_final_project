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
from keras import utils as np_utils

train_df = pd.read_csv("data/processed_full_training_dataset.csv", sep=',', names=['Sentiment', 'Phrase'])
df_test = pd.read_csv("data/processed_iphone.csv", sep=",",names=['Sentiment', 'Phrase'], encoding='utf-8',header=None)
raw_docs_train = train_df['Phrase'].values
raw_test = df_test['Phrase'].values
#print raw_docs_train
print raw_test
train_df['Sentiment'] = train_df['Sentiment'].map({'positive': 0, 'negative': 1, 'neutral': 2})
sentiment_train = train_df['Sentiment'].values

num_labels = len(np.unique(sentiment_train))
print 'num_labels: {}'.format(num_labels)

# text pre-processing
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])
stemmer = SnowballStemmer('english')

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

# if is_lstm:
#     return word_id_train, y_train_enc, dictionary_size, num_labels, word_id_test
# else:
#     return word_id_train, sentiment_train, word_id_test
