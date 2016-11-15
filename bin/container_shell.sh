DATA_DIR=$(pwd)/data
echo "mounting data in "$DATA_DIR

docker run -it -v $DATA_DIR:/data vgg_docker:latest /bin/bash
