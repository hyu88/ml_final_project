

def LSTM_classifier(word_id_train,y_train_enc,dictionary_size,num_labels,num_epochs,layer):
    # import library
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Embedding
    from keras.layers import LSTM, Bidirectional
    import matplotlib.pyplot as plt
    from keras import optimizers
    # LSM
    print "fitting LSTM ..."
    model = Sequential()
    #dictionary_size:output_dim, 128:input_dim
    model.add(Embedding(dictionary_size, 128,embeddings_initializer='uniform', dropout=0.2))
    #model.add(Bidirectional(LSTM(layer, return_sequences=True)))
    #model.add(Bidirectional(LSTM(layer)))

    #128 batch_size, droput_W, dropout_U
    #dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
    #dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections
    model.add(LSTM(layer, dropout_W=0.2, dropout_U=0.2))
    model.add(Dense(num_labels))
    #Dense implements the operation: output = activation(dot(input, kernel) + bias)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history =model.fit(word_id_train, y_train_enc,validation_split=0.33, nb_epoch=num_epochs, batch_size=256, verbose=1)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    loss_and_metrics = model.evaluate(word_id_train, y_train_enc, batch_size=256)
    print 'loss: {}, training accuracy with LSTM: {}'.format(loss_and_metrics[0], loss_and_metrics[1])

    return model