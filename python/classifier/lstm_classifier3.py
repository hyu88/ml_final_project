MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def LSTM_classifier(word_id_train,y_train_enc,num_labels,num_epochs,layer,num_words,embedding_matrix):
    # import library
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Embedding
    from keras.layers import LSTM, Bidirectional
    from keras import optimizers
    from keras.layers import Flatten


    # LSTM
    print "fitting LSTM ..."
    model = Sequential()

    print 'embedding layer .....'
    print 'num_words {}, embedding_dim {}, embedding_matrix {}, max_squenence_length {}: '.format(num_words,EMBEDDING_DIM,embedding_matrix,MAX_SEQUENCE_LENGTH)
    print 'embedding_matrix: {}/{}'.format(len(embedding_matrix),len(embedding_matrix[0]))
    print 'embedding done....'
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Training model.')

    model.add(embedding_layer)
   # model.add(Flatten())

    #model.add(Bidirectional(LSTM(layer, return_sequences=True)))
    #model.add(Bidirectional(LSTM(layer)))

    model.add(LSTM(layer,   dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(num_labels))
    adam=optimizers.Adam(lr=0.02, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #model.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=1e-08, decay=0.0)
    #Dense implements the operation: output = activation(dot(input, kernel) + bias)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print (model.summary())
    model.fit(word_id_train, y_train_enc, nb_epoch=num_epochs, batch_size=256, verbose=1)
    loss_and_metrics = model.evaluate(word_id_train, y_train_enc, batch_size=256)
    print 'loss: {}, training accuracy with LSTM: {}'.format(loss_and_metrics[0], loss_and_metrics[1])

    return model