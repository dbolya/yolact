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
	--cap-add=SYS_PTRACE \
	--security-opt=seccomp:unconfined \
	--security-opt=apparmor:unconfined \
  --gpus all \
  --user $(id -u) \
  --workdir /home/ros \
  --mount type=bind,source="${yoldact_docker}/home/ros",target=/home/ros \
  --mount type=bind,source="${yoldact_docker}/usr/local/lib/python3.8/dist-packages",target=/usr/local/lib/python3.8/dist-packages \
  --name yolact-desktop-nvidia \
  --security-opt apparmor:unconfined \
  --net=host \
  --env="DISPLAY" \
  --volume="$HOME/.Xauthority:/home/ros/.Xauthority:rw" \
  --entrypoint bash \
  yolact-desktop:v0.1