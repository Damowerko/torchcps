#!/bin/bash
set -e
REGISTRY="lc1:32000"
DOCKER_USER="damowerko"
IMAGE_NAME="torchcps"
IMAGE_TAG="latest"
DOCKER_URI="$REGISTRY/$DOCKER_USER/$IMAGE_NAME:$IMAGE_TAG"
export DOCKER_BUILDKIT=1
docker pull $DOCKER_URI || true
docker build . -f ./docker/Dockerfile -t $DOCKER_URI --build-arg BUILDKIT_INLINE_CACHE=1 
docker push $DOCKER_URI
