#!/bin/bash
docker build -t yolact-base:v0.1 --build-arg BUILD_USER_ID=$(id -u) --build-arg BUILD_GROUP_ID=$(id -g) dockerfiles/base
docker build -t yolact-workspace:v0.1 dockerfiles/workspace
docker build -t yolact-desktop:v0.1 dockerfiles/desktop