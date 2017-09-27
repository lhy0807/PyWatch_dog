# -*- coding: cp936 -*-
from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.layers.wrappers import *
import numpy as np

def model():
    model = Sequential()

    model.add(BatchNormalization(batch_input_shape=(32, 10, 66, 200, 3)))

    model.add(TimeDistributed(Conv2D(3, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu')))
    model.add(TimeDistributed(Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu')))
    model.add(TimeDistributed(Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu')))

    model.add(TimeDistributed(Conv2D(48, kernel_size=(3,3), padding='valid', activation='relu')))
    model.add(TimeDistributed(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu')))

    model.add(Dropout(0.5))
    model.add(TimeDistributed(Flatten()))

    model.add(TimeDistributed(Dense(1164, activation='relu')))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(LSTM(100,
                 return_sequences=True,
                 batch_input_shape=(32, 10, 100),
                 stateful=True))
    model.add(LSTM(100,
                 return_sequences=True,
                 stateful=True))
    model.add(TimeDistributed(Dense(50, activation='relu')))
    model.add(TimeDistributed(Dense(10, activation='relu')))
    model.add(Dense(9, activation='softmax'))

    model.summary()

    return model
