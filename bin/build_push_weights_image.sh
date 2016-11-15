# make sure you have run "docker login" before

docker build -f weights/Dockerfile_vgg_weights -t dominicbreuker/vgg_weights:latest weights
docker push dominicbreuker/vgg_weights
