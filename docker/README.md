# Getting environment up and running

## Set up local development environment

To get the a local development environment up and running quickly using docker run:
```bash
./bin/build-images.sh
./bin/set-me-up.sh
```

You can skip running `./bin/build-images.sh` if the images are already built.

The `./bin/set-me-up.sh` script will do the following:
- Run a docker container
- Copy the built workspace to the destination folder of choice
- Stop the container

## Use computer workspace on docker

If you have your workspace locally and want to use the docker machine to run your code do:
```bash
./bin/run-workspace.sh <ABSOLUTE_PATH_OF_FOLDER_PROVIDED_IN_SET_ME_UP>
```

and then **inside the running container bash terminal** run:
```bash
/usr/bin/lxpanel --profile LXDE
````
