from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.layers.wrappers import *
import numpy as np

def model():
    conv_model = Sequential()

    conv_model.add(BatchNormalization(input_shape=(66,200,3)))

    conv_model.add(Conv2D(3, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    conv_model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
    conv_model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))

    conv_model.add(Conv2D(48, kernel_size=(3,3), padding='valid', activation='relu'))
    conv_model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))

    conv_model.add(Dropout(0.5))
    conv_model.add(Flatten())

    conv_model.add(Dense(1164, activation='relu'))
    conv_model.add(Dense(100, activation='relu'))
    conv_model.add(Dense(50, activation='relu'))
    conv_model.add(Dense(10, activation='relu'))
    conv_model.add(Dense(9, activation='softmax'))

    conv_model.summary()
    return conv_model
