from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from grabscreen import grab_screen
from getkeys import key_check
import numpy as np
from directkeys import PressKey, ReleaseKey, W, A, S, D
import time, random, cv2

model = Sequential()

model.add(BatchNormalization(input_shape=(66,200,3)))

model.add(Conv2D(3, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2), padding='valid', activation='relu'))

model.add(Conv2D(48, kernel_size=(3,3), padding='valid', activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), padding='valid', activation='relu'))

model.add(Dropout(0.5))
model.add(Flatten())

model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(9, activation='softmax'))

model.summary()
model.load_weights('cnn_2d.h5')

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
##    ReleaseKey(S)


def right():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)


def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def main():

    paused = False
    screen = grab_screen(region = (0, 40, 1024, 768))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

    mode_choice = 0

    while(True):
        if not paused:
            screen = grab_screen(region = (0, 40, 1024, 768))
            screen = cv2.resize(screen, (200,66))

            prediction = model.predict(np.expand_dims(screen,axis=0))[0]
            #prediction = np.array(prediction) * np.array([4.5, 0.1, 0.1, 0.1, 1.8, 1.8, 0.5, 0.5, 0.2])

            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'
            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'
            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                no_keys()
                choice_picked = 'nokeys'

            print('Choice: {0}'.format(choice_picked))

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

if __name__ == "__main__":
    main()
