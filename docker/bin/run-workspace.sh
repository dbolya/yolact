#!/bin/bash
mode=$1
if [ -z "$1" ]
  then
    mode="it"
fi
docker container run --rm -$mode \
  --user $(id -u) \
  --name yolact-workspace \
  --workdir /home/ros/workspace \
  yolact-workspace:v0.1 \
  bash
