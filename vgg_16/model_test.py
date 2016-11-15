import numpy as np
import os
from keras.optimizers import SGD
from model import vgg_16, load_image_vgg_16
from functools32 import lru_cache

SUCCESS_STRING = 'Asserted {} is a {} --> True'
ERROR_STRING = 'Asserted {} is a {} --> False \
    There seems to be a problem with loading model and weights'


def test_pretrained_weights():
    model = vgg_16()
    test_image_paths = load_test_image_paths()
    for image_path in test_image_paths:
        image = load_image_vgg_16(image_path)
        prediction = model.predict(image)
        top5 = np.argsort(prediction)[0][::-1][0:5]
        print("Predictions for {}:".format(os.path.split(image_path)[1]))
        for class_label_index in top5:
            class_label = get_label_by_index(class_label_index)
            print("{} - {}".format(class_label_index, class_label))
        assert_one_good_label_in_top_5(image_path, top5)
    return None


def assert_one_good_label_in_top_5(image_path, top5):
    image_name = os.path.split(image_path)[1]
    if image_name == 'cat1.jpg':
        assert_category_in_top5(image_name, 287, top5)
    elif image_name == 'cat2.jpg':
        assert_category_in_top5(image_name, 223, top5)
    elif image_name == 'dog1.jpg':
        assert_category_in_top5(image_name, 211, top5)
    elif image_name == 'dog2.jpg':
        assert_category_in_top5(image_name, 235, top5)
    elif image_name == 'ipod.jpg':
        assert_category_in_top5(image_name, 605, top5)
    else:
        raise 'Unknown test image'


def assert_category_in_top5(image_name, category, top5):
    category_name = get_label_by_index(category)
    assert category in top5, ERROR_STRING.format(image_name, category_name)
    print(SUCCESS_STRING.format(image_name, category_name))


def load_test_image_paths():
    image_paths = []
    for filename in os.listdir(test_image_directory()):
        if filename.endswith(".jpg"):
            image_paths.append(os.path.join(test_image_directory(), filename))
    return image_paths


@lru_cache(maxsize=1)
def load_class_labels():
    labels_path = os.path.join(current_directory(), "synset_words.txt")
    class_labels = []
    with open(labels_path, "r") as lines:
        for line in lines:
            class_labels.append(line.rstrip("\n").split(" ", 1)[1])
    return class_labels


def get_label_by_index(index):
    return load_class_labels()[index]


def current_directory():
    return os.path.dirname(os.path.abspath(__file__))


def test_image_directory():
    return os.path.join(current_directory(), 'test_images')


if __name__ == "__main__":
    test_pretrained_weights()
