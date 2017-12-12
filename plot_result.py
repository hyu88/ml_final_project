import matplotlib as plt
import numpy as np

import matplotlib as plt

num_epochs = 3
epoch = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
loss = []
test_acc = []
val_acc = []
for num_epochs in epoch:
    is_lstm = True
    from classifier import lstm_classifier

    clf = lstm_classifier.LSTM_classifier(trainX, trainY, dictionary, num_labels, num_epochs)
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
plt.plot(epoch, loss, linewidth=2.0, color='green')
plt.plot(epoch, test_acc, linewidth=2.0, color='blue')
plt.plot(epoch, val_acc, linewidth=2.0, color='red')
plt.legend(['loss', 'test_acc', 'val_acc'], loc='upper left')
plt.show()

