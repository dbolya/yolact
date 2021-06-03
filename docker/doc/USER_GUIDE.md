# Requirements

This is a tutorial that explains how to run `TSDF-plusplus` using Docker
so that you can easily and quickly get up and running productively.

This tutorial assumes you set up your development environment as described in [Docker README.md](../docker/README.md)

Specifically, this tutorial assumes you have your computer able to run: `./bin/docker-develop.sh <ABSOLUTE_PATH_OF_CATKIN_WS>`
in docker folder. Once in the docker machine, you will require to be able to run a successful build:
```
catkin build tsdf_plusplus_ros rgbd_segmentation mask_rcnn_ros cloud_segmentation
```
This should run without problem if the setup is successful.

Additionally, it also requires that you are able to start the desktop in case you wish to visualize
results using rviz. For that, you should be able to run on the docker machine:
```bash
/usr/bin/lxpanel --profile LXDE
```

## Voxblox verification

The first step will be to launch `Voxblox`, the core mapping framework `TSDF-plusplus` relies on.
See [Voxblox documentation](https://voxblox.readthedocs.io/en/latest/index.html) for details.

First step, which is common thoughout this tutorial is to run `./bin/docker-develop.sh <ABSOLUTE_PATH_OF_CATKIN_WS> nvidia-docker`.
Once you are in the docker machine, launch the desktop by running:
```bash
/usr/bin/lxpanel --profile LXDE
```

Then go to the main menu with your mouse and click one one of the terminals.
I recomend using XXXX.

From that terminal run:
```bash
mkdir /home/ros/catkin_ws/bags
wget http://robotics.ethz.ch/~asl-datasets/iros_2017_voxblox/data.bag /home/ros/catkin_ws/bags
sed -i 's/\/path\/to\/data.bag/\/home\/ros\/catkin_ws\/bags\/data.bag/g' /home/ros/catkin_ws/src/voxblox/voxblox_ros/launch/cow_and_lady_dataset.launch
```

These lines will download the dataset and keep in the mounted volume, thus it will as well get stored on your computer. This way,
the next time you launch the `docker` image, you won't need to download it again.

On the same terminal run the `ros` daemon:
```bash
source devel/setup.bash
roscore
```

On another window run `rviz` to visualize the results:
```bash
rosrun rviz rviz -d rviz/asl-dataset.rviz
```

Finally, on a third terminal run the actual `voxblox`:
```bash
source devel/setup.bash
roslaunch voxblox_ros cow_and_lady_dataset.launch
```

You should see in rviz after a while the TSDF-integrated shape in rviz, which should consist of a lady as well as a cow in a room.

Finally, to manage multiple terminal I highly recommend using [tmux](https://github.com/tmux/tmux/wiki/Getting-Started)
