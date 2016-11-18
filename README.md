# VGG 16 inside Docker
Easily extract image features from various layers of VGG16 with this Docker image.
Or just use it in prediction mode to get labels for input images.

The Docker image contains a pre-trained VGG16 model along with scripts to load images from a directory and to extract features from them.

Run `docker run -it --rm -v $DATA_DIR:/data -v $OUTPUT_DIR:/output dominicbreuker/vgg_docker:latest python /vgg_16/extractor.py` to extract features from all images in `$DATA_DIR`.
Resuts will be written to `$OUTPUT_DIR`.

You can pass various arguments to `extractor.py`:
- `--mode` (`-m`) defines the kind of feature you want to generate. 4 modes are available:
  - `label`: returns the prediction in plain english (e.g, 'ipod'). You can see a list of labels in `/vgg_16/synset_words.txt`.
  - `softmax`: returns a 1000-dim vector with class probabilities for each of the 1000 possible ImageNet labels.
  - `dense`: returns a 4096-dim vector extracted from the layer immediately before softmax.
  - `convolutional`: returns the output of the last convolutional layer
- `--height` (`-hs`) and `--width` (`-ws`) define the image size. Default is 256x256. Can only be changed if in `convolutional` mode. Otherwise, images must be resized to 256x256 like it was done in the ImageNet competition.
- `--extension` (`-e`) defines the files you can look for in `$DATA_DIR`. Defaults to `jpg`. The script will process all images with the given extension anywhere in the file tree below `$DATA_DIR`.

Defaults for each mode are as follows:
`docker run -it --rm -v $DATA_DIR:/data -v $OUTPUT_DIR:/output dominicbreuker/vgg_docker:latest python /vgg_16/extractor.py -m label -hs 256 -ws 256 -e jpg`

After running this script, you will find the following two files in `$OUTPUT_DIR`:
- `image_files_vgg16_label_256x256_<timestamp>.npz` with a list of image file names (your IDs)
- `extractions_vgg16_label_256x256_<timestamp>.npz` with a list of features

## VGG Background

VGG is a pre-trained CNN created using the ImageNet dataset.
Read [this paper](https://arxiv.org/pdf/1409.1556.pdf) for details regarding the model.
Or check out the [ILSVRC 2014 results](http://image-net.org/challenges/LSVRC/2014/results) to see that it made 1st place in the competition.
Must be a good one ;)
You can use it to build powerful image processing tools by transferring the knowledge within the model to your application, like described [here](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).

## How it is built

### Weights

The Docker image contains pre-trained weights taken from this [Gist](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).
These weights are a direct transformation of the original authors' Caffe model.
They are stored in `/weights/vgg16_weights_tensorflow.h5`.

## Image pre-processing

The script will pre-process images in the same way it was done during creation of VGG16.
The steps include subtraction of mean pixel values based on ImageNet (seperatly by color) and cropping out borders of the image (e.g., of 256x256 image, only the center 224x224 pixels are retained).

## Tests

To see if you are using the weights correctly, check out `/vgg_16/model_test.py`.
It will predict the top5 class labels for each `/vgg_16/test_images/*.jpg`.
This script is run during the Docker image build to verify predictions are reasonable.

### Sources of test images:
- cat1.jpg: [Dwight Sipler](http://www.flickr.com/people/62528187@N00) from Stow, MA, USA, [Gillie hunting (2292639848)](https://commons.wikimedia.org/wiki/File:Gillie_hunting_(2292639848).jpg), [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/legalcode)
- cat2.jpg: The original uploader was [DrL](https://en.wikipedia.org/wiki/User:DrL) at [English Wikipedia](https://en.wikipedia.org/wiki/) [Blackcat-Lilith](https://commons.wikimedia.org/wiki/File:Blackcat-Lilith.jpg), [CC BY-SA 2.5
](https://creativecommons.org/licenses/by-sa/2.5/legalcode)
- dog1.jpg: HiSa Hiller, Schweiz, [Thai-Ridgeback](https://commons.wikimedia.org/wiki/File:Thai-Ridgeback.jpg), [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/legalcode)
- dog2.jpg: [Military dog barking](https://commons.wikimedia.org/wiki/File:Military_dog_barking.JPG), in the [public domain](https://en.wikipedia.org/wiki/public_domain)
- ipod.jpg: [Marcus Quigmire](http://www.flickr.com/people/41896843@N00) from Florida, USA, [Baby Bloo taking a dip (3402460462)](https://commons.wikimedia.org/wiki/File:Baby_Bloo_taking_a_dip_(3402460462).jpg), [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/legalcode)
