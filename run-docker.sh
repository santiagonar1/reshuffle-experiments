#!/usr/bin/env bash

DOCKER_IMAGE=santiagonar1/cpp-dev:latest
FOLDER_NAME=code

docker run --rm -it -v .:/${FOLDER_NAME} -w /${FOLDER_NAME} ${DOCKER_IMAGE} bash
