from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np
import cnn_2d

###Parameters###

batch_size = 32
epochs = 100
packages = 36
weight_file = 'cnn_2d.h5'
#Different fragments: 36,73,110,147,184,221,259,296,333,370

###Parameters###

model = cnn_2d.model()

if os.path.isfile(weight_file):
    model.load_weights(weight_file)


datagen = ImageDataGenerator(width_shift_range=0.05,
                             height_shift_range=0.05,
                             zoom_range=0.05,
                             rotation_range=10,)

raw_x = []
raw_y = []

for i in range(packages):
    i = i+1
    filename = ''.join(['./data/',str(i),'.npy'])    
    dataset = np.load(filename)

    for n in range(500):
        raw_x.append(dataset[n][0])
        raw_y.append(dataset[n][1])
        
    dataset = []

    print('File ',str(i),' loaded')

raw_x = np.array(raw_x)
raw_y = np.array(raw_y)

model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-04), metrics=['categorical_accuracy'])

x_train, x_val, y_train, y_val = train_test_split(
    raw_x, raw_y, test_size=0.1)

raw_x = []
raw_y = []

checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='auto')
tb = TensorBoard()

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(x_train) / batch_size,
                    epochs=epochs,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpoint, tb])

model.save_weights(weight_file)
