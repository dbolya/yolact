#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --no-requeue

# Usage: ./resume.sh config batch_size resume_file

module load python/3.6.4_gcc5_np1.14.5
module load cuda/9.0

cd $SCRATCH/yolact

python3 train.py --config $1 --batch_size $2 --resume=$3 --save_interval 5000 --start_iter=-1 >>logs/$1_log 2>&1
