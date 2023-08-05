import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import Dense, Dropout, Flatten
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
from tensorflow.keras import regularizers

tic = time.perf_counter()
plt.rc('font', size=18)
# plt.rcParams['figure.constrained_layout.use'] = True

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
n = 50000
weight = 0.0001
x_train = x_train[1:n]
y_train = y_train[1:n]
# x_test=x_test[1:500]; y_test=y_test[1:500]

with tf.device('/device:GPU:1'):
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    print("orig x_train shape:", x_train.shape)
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    use_saved_model = False
    if use_saved_model:
        model = keras.models.load_model("cifar.model")
    else:
        model = keras.Sequential()
        '''model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(weight)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.summary()'''
        '''model = keras.Sequential()
		model.add(Conv2D(16, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
		model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
		model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))
		model.add(Flatten())
		model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(weight)))
		model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
		model.summary()'''
        model.add(Conv2D(8, (3, 3), padding='same', input_shape=x_train.shape[1:], activation='relu'))
        model.add(Conv2D(8, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), strides=(2, 2), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(weight)))
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
        model.summary()
    batch_size = 128
    epochs = 60
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    model.save("cifar.model")
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Test loss: % f accuracy: % f" % (score[0], score[1]))
    score = model.evaluate(x_train, y_train, verbose=0)
    print("Train loss: % f accuracy: % f" % (score[0], score[1]))
    fig = plt.figure()
    plt.subplot(211)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    toc = time.perf_counter()
    text = 'n = ' + str(n // 1000) + 'K; ' + '%.2f' % (toc - tic) + ' sec'
    # text = 'Weight = ' + str(weight) + '; ' + '%.2f' % (toc - tic) + ' sec'
    # text = '%.2f' % (toc - tic) + ' sec'
    fig.text(0, 0.03, text)
    plt.show()
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(x_train, y_train)

preds = model.predict(x_train)
y_pred = np.argmax(preds, axis=1)
y_train1 = np.argmax(y_train, axis=1)
print(classification_report(y_train1, y_pred))
print(confusion_matrix(y_train1, y_pred))

preds = model.predict(x_test)
y_pred = np.argmax(preds, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(classification_report(y_test1, y_pred))
print(confusion_matrix(y_test1, y_pred))

preds = dummy.predict(x_test)
y_pred = np.argmax(preds, axis=1)
y_test1 = np.argmax(y_test, axis=1)
print(classification_report(y_test1, y_pred))
print(confusion_matrix(y_test1, y_pred))
