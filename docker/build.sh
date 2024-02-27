#!/bin/bash
set -e
export DOCKER_BUILDKIT=1
docker pull docker.io/damowerko/motion-planning:latest || true
docker build . -f ./docker/Dockerfile -t docker.io/damowerko/motion-planning --build-arg BUILDKIT_INLINE_CACHE=1 
docker push docker.io/damowerko/motion-planning
