#!/bin/bash -l

#SBATCH --job-name=baidu-features
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --partition=cpu
#SBATCH --array=0-1000%32

source ${HOME}/.bashrc
mamba activate baidu-ultr

python features.py train_part=$SLURM_ARRAY_TASK_ID
