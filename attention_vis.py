import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import *
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard
from vis.visualization import visualize_saliency
from vis.utils import utils
from grabscreen import grab_screen
import cv2


# Build the network with weights
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
model.add(Dense(9, activation='softmax', name='predictions'))

model.summary()
model.load_weights('cnn_2d.h5')
print('Model loaded.')

# The name of the layer we want to visualize
layer_name = 'predictions'
layer_idx = [idx for idx, layer in enumerate(model.layers) if layer.name == layer_name][0]

plt.axis('off')
plt.ion()
data = np.load('./data/100.npy')
for i in range(500):
    heatmaps = []
    screen = data[i][0]
    seed_img = screen
    x = np.expand_dims(img_to_array(seed_img), axis=0)
    x = preprocess_input(x)
    pred_class = np.argmax(model.predict(x))

    # Here we are asking it to show attention such that prob of `pred_class` is maximized.
    heatmap = visualize_saliency(model, layer_idx, [pred_class], seed_img)
    heatmaps.append(heatmap)

    plt.imshow(utils.stitch_images(heatmaps))
    plt.title('Saliency map')
    plt.pause(0.0001)
