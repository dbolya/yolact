#!/bin/bash
#SBATCH -p GPU-small
#SBATCH -t 2:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --no-requeue

# Usage: ./eval.sh weights extra_args

module load python/3.12
module load cuda/9.0

cd $SCRATCH/yolact

yolact-eval --trained_model=$1 --no_bar $2 > logs/eval/$(basename -- $1).log 2>&1
