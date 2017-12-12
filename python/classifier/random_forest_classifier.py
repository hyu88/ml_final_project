from sklearn.metrics import accuracy_score

def RF_classifier(word_id_train,sentiment_train,max_depth=8):

# Random Forest
    print 'fitting Random Forest'
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(max_depth=10, random_state=0)
    clf.fit(word_id_train,sentiment_train)
    pred=clf.predict(word_id_train)
    acc = accuracy_score(sentiment_train, pred)
    print 'training accuracy with Random Forest: {}'.format(acc)
    return clf