from sklearn.metrics import accuracy_score

def Gauss_bayes_classifier(word_id_train,sentiment_train):

# naive bayes
    print 'fitting naive bayes'
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(word_id_train,sentiment_train)
    pred=clf.predict(word_id_train)
    acc=accuracy_score(sentiment_train,pred)
    print 'training accuracy with Gauss Bayes: {}'.format(acc)

    return clf

