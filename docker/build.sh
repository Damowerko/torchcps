#!/bin/bash
set -e
IMAGE_NAME="torchcps"
export DOCKER_BUILDKIT=1
docker pull docker.io/damowerko/$IMAGE_NAME:latest || true
docker build . -f ./docker/Dockerfile -t docker.io/damowerko/$IMAGE_NAME --build-arg BUILDKIT_INLINE_CACHE=1 
docker push docker.io/damowerko/$IMAGE_NAME
