#!/bin/bash -l

#SBATCH --job-name=baidu-ultr
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --mem 120GB
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.k.hager@uva.nl

source ${HOME}/.bashrc
conda activate baidu-ultr

python main.py
