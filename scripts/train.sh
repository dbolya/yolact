#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gres gpu:p100:1

# Usage: ./train.sh config batch_size

module add python/3.6.4_gcc5_np1.14.5
module add cuda/9.2

cd $SCRATCH/yolact

python3 train.py --config $1 --batch_size $2 --save_interval 5000 &>logs/$1_log
