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

source ${HOME}/.bashrc
conda activate baidu-ultr

python main.py +data_type=val
