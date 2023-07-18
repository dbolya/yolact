#!/bin/bash

start=`date +%s`
trainfile="train2017.zip"
valfile="val2017.zip"
annotations_trainval2014="annotations_trainval2014.zip"
annotations_trainval2017="annotations_trainval2017.zip"

# handle optional download dir
if [ -z "$1" ]
  then
    # navigate to ./data
    echo "navigating to ./data/ ..."
    mkdir -p ./data
    cd ./data/
    mkdir -p ./coco
    cd ./coco
    mkdir -p ./images
    mkdir -p ./annotations
  else
    # check if specified dir is valid
    if [ ! -d $1 ]; then
        echo $1 " is not a valid directory"
        exit 0
    fi
    echo "navigating to " $1 " ..."
    cd $1
fi

if [ ! -d images ]
  then
    mkdir -p ./images
fi

# Download the image data.
cd ./images
# check train2017.zip exist
if [ ! -f "$trainfile" ]
  then
  echo "Downloading MSCOCO train images ..."
  curl -LO http://images.cocodataset.org/zips/train2017.zip
  else
  echo "train2017.zip exist"
fi

# check val2017.zip exist
if [ ! -f "$valfile" ]
  then
  echo "Downloading MSCOCO val images ..."
  curl -LO http://images.cocodataset.org/zips/val2017.zip
  else
  echo "val2017.zip exist"
fi

cd ../
if [ ! -d annotations ]
  then
    mkdir -p ./annotations
fi

# Download the annotation data.
cd ./annotations

if [ ! -f "$annotations_trainval2014" ] || [ ! -f "$annotations_trainval2017" ]; then
    echo "Downloading MSCOCO train/val annotations ..."

    if [ ! -f "$annotations_trainval2014" ]; then
        echo "$annotations_trainval2014 not exist, now downloading ..."
        curl -LO http://images.cocodataset.org/annotations/annotations_trainval2014.zip
    fi

    if [ ! -f "$annotations_trainval2017" ]; then
        echo "$annotations_trainval2017 not exist, now downloading..."
        curl -LO http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    fi
else
    echo "annotations exist"
fi

echo "Finished downloading. Now extracting ..."

# Unzip data
if ls ../images/*.jpg 1> /dev/null 2>&1; then
  echo "jpg files exist, starting deletion..."
  rm ../images/*.jpg
fi
echo "Extracting train images ..."
unzip -qqjd ../images ../images/train2017.zip
echo "Extracting val images ..."
unzip -qqjd ../images ../images/val2017.zip

if ls ./annotations/*.json 1> /dev/null 2>&1; then
  echo "json files exist, starting deletion..."
  rm ./annotations/*.json
fi
echo "Extracting annotations ..."
unzip -qqd .. ./annotations_trainval2014.zip
unzip -qqd .. ./annotations_trainval2017.zip

echo "Download completed, Removing zip files ..."
rm ../images/train2017.zip
rm ../images/val2017.zip
rm ./annotations_trainval2014.zip
rm ./annotations_trainval2017.zip

end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"
