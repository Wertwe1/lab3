#!/bin/bash
#SBATCH --job-name=bert_mlm
#SBATCH --output=logs/config1/config1_%A_%a.out
#SBATCH --error=logs/config1/config1_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=GPU-shared         # Use the GPU partition
#SBATCH --gpus=v100-32:1


module purge
source ~/miniforge3/etc/profile.d/conda.sh
conda activate lab3

# 3) run your training
python run_train.py --config configs/config1.yaml
