DATA_DIR=$(pwd)/data
mkdir -p $DATA_DIR
echo "mounting data in "$DATA_DIR

OUTPUT_DIR=$(pwd)/output
mkdir -p $OUTPUT_DIR
echo "mounting output dir in "$OUTPUT_DIR

docker run -it --rm -v $DATA_DIR:/data -v $OUTPUT_DIR:/output vgg_docker:latest python /vgg_16/extractor.py -m label -hs 256 -ws 256 -e jpg
