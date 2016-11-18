from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, \
    ZeroPadding2D
from PIL import Image
import os
import numpy as np
from functools32 import lru_cache
import tensorflow as tf
tf.python.control_flow_ops = tf


def vgg_16(include_layers="softmax", input_size=(256, 256), load_weights=True):
    target_size = cropped_image_size(input_size)
    if include_layers == "convolutional":
        layers = convolutional_layers(target_size)
    elif include_layers == "dense":
        layers = convolutional_layers(target_size) + dense_layers()
    elif include_layers == "softmax":
        layers = convolutional_layers(target_size) + dense_layers() + \
                 prediction_layer()
    else:
        raise Exception("Unsupported model version: {}".format(include_layers))

    model = Sequential(layers)

    if load_weights:
        model.load_weights(weights_path(), by_name=True)

    return model


def convolutional_layers(target_size):
    conv_layers_1 = [
        ZeroPadding2D((1, 1), input_shape=(3, target_size[0], target_size[1]),
            name='zero_padding_1_1'),
        Convolution2D(64, 3, 3, activation='relu', name='convolutional_1_1'),
        ZeroPadding2D((1, 1), name='zero_padding_1_2'),
        Convolution2D(64, 3, 3, activation='relu', name='convolutional_1_2'),
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_1')
    ]

    conv_layers_2 = [
        ZeroPadding2D((1, 1), name='zero_padding_2_1'),
        Convolution2D(128, 3, 3, activation='relu', name='convolutional_2_1'),
        ZeroPadding2D((1, 1), name='zero_padding_2_2'),
        Convolution2D(128, 3, 3, activation='relu', name='convolutional_2_2'),
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_2')
    ]

    conv_layers_3 = [
        ZeroPadding2D((1, 1), name='zero_padding_3_1'),
        Convolution2D(256, 3, 3, activation='relu', name='convolutional_3_1'),
        ZeroPadding2D((1, 1), name='zero_padding_3_2'),
        Convolution2D(256, 3, 3, activation='relu', name='convolutional_3_2'),
        ZeroPadding2D((1, 1), name='zero_padding_3_3'),
        Convolution2D(256, 3, 3, activation='relu', name='convolutional_3_3'),
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_3')
    ]

    conv_layers_4 = [
        ZeroPadding2D((1, 1), name='zero_padding_4_1'),
        Convolution2D(512, 3, 3, activation='relu', name='convolutional_4_1'),
        ZeroPadding2D((1, 1), name='zero_padding_4_2'),
        Convolution2D(512, 3, 3, activation='relu', name='convolutional_4_2'),
        ZeroPadding2D((1, 1), name='zero_padding_4_3'),
        Convolution2D(512, 3, 3, activation='relu', name='convolutional_4_3'),
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_4')
    ]

    conv_layers_5 = [
        ZeroPadding2D((1, 1), name='zero_padding_5_1'),
        Convolution2D(512, 3, 3, activation='relu', name='convolutional_5_1'),
        ZeroPadding2D((1, 1), name='zero_padding_5_2'),
        Convolution2D(512, 3, 3, activation='relu', name='convolutional_5_2'),
        ZeroPadding2D((1, 1), name='zero_padding_5_3'),
        Convolution2D(512, 3, 3, activation='relu', name='convolutional_5_3'),
        MaxPooling2D((2, 2), strides=(2, 2), name='max_pool_5'),
        Flatten(name='flatten')
    ]

    return conv_layers_1 + \
        conv_layers_2 + \
        conv_layers_3 + \
        conv_layers_4 + \
        conv_layers_5


def dense_layers():
    dense_layers = [
        Dense(4096, activation='relu', name='dense_1'),
        Dropout(0.5),
        Dense(4096, activation='relu', name='dense_2'),
        Dropout(0.5)
    ]
    return dense_layers


def prediction_layer():
    return [Dense(1000, activation='softmax', name='softmax')]


def load_image_vgg_16(image_path, image_size=(256, 256)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize(image_size)
    image = np.asarray(image, dtype='float32')
    image = rgb_to_bgr(image)
    return preprocess_image(image, cropped_image_size(image_size))


def rgb_to_bgr(image):
    image[:, :, [0, 1, 2]] = image[:, :, [2, 1, 0]]
    return image


def preprocess_image(image, crop_size):
    image[:, :, 0] -= 123.68
    image[:, :, 1] -= 116.779
    image[:, :, 2] -= 103.939
    image = image.transpose((2, 0, 1))
    image = crop_image(image, crop_size)
    image = np.expand_dims(image, axis=0)
    return image


def crop_image(img, crop_size):
    img_size = img.shape[1:]
    img = img[:, (img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2,
                 (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]
    return img


# orignal VGG_16 used (256, 256) images cropped to (224, 224)
# we apply the same cropping during preprocessing
CROP_FACTOR = 0.875


def cropped_image_size(image_size):
    return (int(CROP_FACTOR * image_size[0]), int(CROP_FACTOR * image_size[1]))


def current_directory():
    return os.path.dirname(os.path.abspath(__file__))


def weights_path():
    return os.path.join(current_directory(), os.pardir, 'weights',
                        'vgg16_weights_tensorflow.h5')


def labels_path():
    return os.path.join(current_directory(), 'synset_words.txt')


@lru_cache(maxsize=1)
def load_class_labels():
    class_labels = []
    with open(labels_path(), "r") as lines:
        for line in lines:
            class_labels.append(line.rstrip("\n").split(" ", 1)[1])
    return class_labels


def get_label_by_index(index):
    return load_class_labels()[index]
