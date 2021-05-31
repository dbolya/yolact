#!/bin/bash
./bin/run-workspace.sh dt
sleep 5
container_id=$(docker ps -aqf "name=yolact-workspace" | tr -d '\n')
echo "Docker container $container_id"
echo "Write absolute path of the folder on your computer, you want the entire '/home/ros/workspace' folder on the docker machine to be copied into:"
read destination_folder
docker cp $container_id:/home/ros/workspace $destination_folder
docker stop -t 1 $container_id