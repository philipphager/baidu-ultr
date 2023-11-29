#!/bin/bash -l

#SBATCH --job-name=baidu-ultr
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64GB
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.k.hager@uva.nl
#SBATCH --array=0-9

MODEL=tencent

source ${HOME}/.bashrc
mamba activate baidu-ultr-features

python main.py \
  train_part=0 \
  data_type=train \
  train_split_id=$SLURM_ARRAY_TASK_ID \
  model="$MODEL" \
  tokens="$MODEL" \
  out_directory=output/"$MODEL"

python main.py \
  data_type=val \
  model="$MODEL" \
  tokens="$MODEL" \
  out_directory=output/"$MODEL"
