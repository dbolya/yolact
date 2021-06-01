#!/bin/bash
./bin/run-workspace.sh dt
sleep 5
container_id=$(docker ps -aqf "name=yolact-workspace" | tr -d '\n')
echo "Docker container $container_id"
echo "Write absolute path of the folder on your computer, you want the work files folders ( /home/ros and /opt/conda ) on the docker machine to be copied into:"
read destination_folder
mkdir -p $destination_folder/home/ros
mkdir -p $destination_folder/opt/conda
docker cp $container_id:/home/ros $destination_folder/home/ros
docker cp $container_id:/opt/conda $destination_folder/opt/conda
docker stop -t 1 $container_id