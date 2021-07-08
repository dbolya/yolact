#! /bin/bash

SCRIPT_DIR=$(dirname "$0")
YOLACT_CONFIG=$1
YOLACT_BATCH_SIZE=$2
YOLACT_CONFIG_FULL="${YOLACT_CONFIG}_config"
YOLACT_LOG_FOLDER="logs/${YOLACT_CONFIG}"
YOLACT_SAVE_FOLDER="weights/${YOLACT_CONFIG}"

# show current configuration and wait for check
echo ============ YOLACT TRAIN ARGS ============
echo "config      : ${YOLACT_CONFIG_FULL}"
echo "batch_size  : ${YOLACT_BATCH_SIZE}"
echo "log_folder  : ${YOLACT_LOG_FOLDER}"
echo "save_folder : ${YOLACT_SAVE_FOLDER}"
echo ===========================================

for i in 5 4 3 2 1
do
   echo "Start in ${i} second(s)"
   sleep 1
done

# change directory and create directories for weight and log
cd ${SCRIPT_DIR}/..

# make a link in the save_folder directory
if [ -f "./${YOLACT_SAVE_FOLDER}/resnet50-19c8e357.pth" ]; then
    echo "Already Initialized for this configuration."
else 
    echo "Initialize for this configuration."
    mkdir -p ./${YOLACT_LOG_FOLDER}
    mkdir -p ./${YOLACT_SAVE_FOLDER}
    ln -s ../resnet50-19c8e357.pth ./${YOLACT_SAVE_FOLDER}/
    ln -s ../resnet101_reducedfc.pth.pth ./${YOLACT_SAVE_FOLDER}/
fi

# start training
python3 train.py \
 --config=${YOLACT_CONFIG_FULL} \
 --log_folder=${YOLACT_LOG_FOLDER} \
 --batch_size=${YOLACT_BATCH_SIZE} \
 --save_folder=${YOLACT_SAVE_FOLDER}/