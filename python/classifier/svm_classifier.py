from sklearn.metrics import accuracy_score

def SVM_classifier(word_id_train, sentiment_train):
# SVM
    print 'fitting SVM'
    from sklearn import svm
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(word_id_train, sentiment_train)
    pred=clf.predict(word_id_train)

    acc=accuracy_score(sentiment_train,pred)
    print 'training accuracy with SVM: {}'.format(acc)

    return clf