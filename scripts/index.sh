#!/bin/bash -l

#SBATCH --job-name=index
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task 16
#SBATCH --mem 28GB
#SBATCH --partition rome
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=p.k.hager@uva.nl

source ${HOME}/.bashrc
mamba activate baidu-ultr-features

python index.py
