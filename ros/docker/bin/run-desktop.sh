#!/bin/bash
yoldact_docker=$1
if [ -z "$1" ]
  then
    echo "Absolute path of the yolact work folder not found"
    exit 1
fi
mode=$2
if [ -z "$2" ]
  then
    mode="it"
fi
xhost +
echo $yoldact_docker
export LIBGL_ALWAYS_INDIRECT=1
nvidia-docker container run --rm -$mode \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  --gpus all \
  --user $(id -u) \
  --workdir /home/ros \
  --mount type=bind,source="${yoldact_docker}/home/ros",target=/home/ros \
  --mount type=bind,source="${yoldact_docker}/opt/conda",target=/opt/conda \
  --name yolact-desktop-nvidia \
  --security-opt apparmor:unconfined \
  --net=host \
  --env="DISPLAY" \
  --volume="$HOME/.Xauthority:/home/ros/.Xauthority:rw" \
  --entrypoint bash \
  yolact-desktop:v0.1