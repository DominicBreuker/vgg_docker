import os
import argparse
import numpy as np
from datetime import datetime
from model import vgg_16, load_image_vgg_16, get_label_by_index


def get_image_files(directory, extension):
    files = []
    print("Looking for '.{}' images in: {}"
          .format(extension, os.path.abspath(directory)))
    for root, directories, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".{}".format(extension)):
                files.append(os.path.join(root, filename))
    print("{} images found...".format(len(files)))
    return files


def data_directory():
    return "/data"


def output_directory():
    return "/output"


def extract_max(model, image_file, image_size):
    softmax = extract_vector(model, image_file, image_size)
    best = np.argsort(softmax)[0][::-1][0]
    return get_label_by_index(best)


def extract_vector(model, image_file, image_size):
    image = load_image_vgg_16(image_file, image_size)
    prediction = model.predict(image)
    return prediction


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extension", nargs="?", type=str,
                        default='jpg',
                        help="Look for images with this extension")
    parser.add_argument("-m", "--mode", nargs="?",
                        choices=['label', 'softmax', 'dense', 'convolutional'],
                        default='label',
                        help="Run in on of these modes. 'Label' for predicting \
                        textual class label, 'softmax' for 1x1000 softmax \
                        vecrors, 'dense' for 1x4096 dense layer output and \
                        'convolutional' for bottleneck features after \
                        convoultional layers")
    parser.add_argument("-hs", "--height", nargs="?", type=int, default=256,
                        help="Image height - can be changed only in \
                        'convolutional' mode")
    parser.add_argument("-ws", "--width", nargs="?", type=int, default=256,
                        help="Image width - can be changed only in \
                        'convolutional' mode")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if (args.mode != 'convolutional') and \
       ((args.height != 256) or (args.width != 256)):
        raise Exception("Custom size can only be used in convolutional mode")

    input_size = (args.height, args.width)
    image_files = get_image_files(data_directory(), args.extension)
    if args.mode == 'label':
        model = vgg_16(include_layers="softmax", input_size=input_size)
        extractor = extract_max
    elif args.mode == 'softmax':
        model = vgg_16(include_layers="softmax", input_size=input_size)
        extractor = extract_vector
    elif args.mode == 'dense':
        model = vgg_16(include_layers="dense", input_size=input_size)
        extractor = extract_vector
    elif args.mode == 'convolutional':
        model = vgg_16(include_layers="convolutional", input_size=input_size)
        extractor = extract_vector
    else:
        raise Exception("unsupported mode: {}".format(args.mode))

    # extract data
    extractions = []
    total_images = len(image_files)
    progress = 0
    for image_file in image_files:
        extractions.append(extractor(model, image_file, input_size))
        progress += 1
        print("{} - Progress: {}/{} ({:3.5f}%) - Image '{}' done ..."
            .format(datetime.now().strftime("%Y.%m.%d %H:%M:%S"),
                    progress, total_images,
                    (float(progress) / float(total_images) * 100),
                    image_file))

    # save results
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    image_files = np.array(image_files)
    image_files_filename = "image_files_vgg16_{}_{}x{}_{}"\
        .format(args.mode, args.height, args.width, current_time)
    np.save(os.path.join(output_directory(), image_files_filename),
            image_files)
    print("Image names saved to {} - shape: {}"
        .format(image_files_filename, image_files.shape))
    print("Load with numpy: 'np.load('{}')'".format(image_files_filename))

    extractions = np.vstack(extractions)
    extractions_filename = "extractions_vgg16_{}_{}x{}_{}"\
        .format(args.mode, args.height, args.width, current_time)
    np.save(os.path.join(output_directory(), extractions_filename),
            extractions)
    print("Extractions saved to {} - shape: {}"
        .format(extractions_filename, extractions.shape))
    print("Load with numpy: 'np.load('{}')'".format(extractions_filename))
