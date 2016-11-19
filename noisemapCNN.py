'''
Created on Nov 18, 2016

@author: Denalex
'''
from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility
import os

os.environ['KERAS_BACKEND'] = "theano"

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from ImageHandler import ImageHandler as im

# how many examples to look at during each training iteration
batch_size = 128
# numbers 0-9, so ten classes
nb_classes = 2
# how many times to run through the full set of examples
nb_epoch = 50

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
# i.e. we will use a n_pool x n_pool pooling window
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

altered800 = 'altered800.npy'
unaltered800 = 'unaltered800.npy'
altered200 = 'altered200.npy'
unaltered200 = 'unaltered200.npy'

training_x = np.concatenate((np.load(altered800), np.load(unaltered800)))
test_x = np.concatenate((np.load(altered200), np.load(unaltered200)))
training_y = np.concatenate((im.get_validation_array(1, 800), im.get_validation_array(0, 800)))
test_y = np.concatenate((im.get_validation_array(1, 200), im.get_validation_array(0, 200)))

training_x = training_x.astype(np.uint8)
test_x = test_x.astype(np.uint8)
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = (training_x, training_y), (test_x, test_y)

# we have to preprocess the data into the right form
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize from [0, 255] to [0, 1]
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                        # apply the filter to only full parts of the image
                        # (i.e. do not "spill over" the border)
                        # this is called a narrow convolution
                        border_mode='valid',
                        # we have a 28x28 single channel (grayscale) image
                        # so the input shape should be (1, 28, 28)
                        input_shape=input_shape))

# we use ReLU, a common and effective convolution activation function
model.add(Activation('relu'))

model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

# then we apply pooling to summarize the features
# extracted thus far
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# compile the model based on backend (Theano or Tensorflow);
# we specify what loss function to optimize, and the optimizer method
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# the training may be slow depending on your computer
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
# how'd we do?
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
