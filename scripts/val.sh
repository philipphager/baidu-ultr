#!/bin/bash -l

#SBATCH --job-name=val
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task 8
#SBATCH --mem 64GB
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.k.hager@uva.nl

echo "Model: $1"
source ${HOME}/.bashrc
mamba activate baidu-ultr-features

python main.py \
  data_type=val \
  model="$1" \
  tokens="$1" \
  out_directory=output/"$1"
