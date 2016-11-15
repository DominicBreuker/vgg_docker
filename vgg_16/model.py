from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
    ZeroPadding2D
from PIL import Image
import os
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf


def vgg_16(load_weights=True):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if load_weights:
        model.load_weights(weights_path())

    return model


def load_image_vgg_16(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((256, 256))
    image = np.asarray(image, dtype='float32')
    image = rgb_to_bgr(image)
    return preprocess_image(image)


def rgb_to_bgr(image):
    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
    return image


def preprocess_image(image):
    image[:, :, 0] -= 123.68
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 103.939
    image = image.transpose((2, 0, 1))
    image = crop_image(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    return image


def crop_image(img, crop_size):
    img_size = img.shape[1:]
    img = img[:, (img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2,
                 (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]
    return img


def current_directory():
    return os.path.dirname(os.path.abspath(__file__))


def weights_path():
    return os.path.join(current_directory(), os.pardir, 'weights',
                        'vgg16_weights_tensorflow.h5')
