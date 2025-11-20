#!/bin/bash
#SBATCH -c 1
#SBATCH -t 12:00:00
#SBATCH --mem=50G

#SBATCH -p kempner
#SBATCH --account kempner_mzitnik_lab
#SBATCH --gres=gpu:1

#SBATCH -o  logs/11_17/initial_training.out
#SBATCH -e logs/11_17/initial_training.err

python -m run.run ../configs/sample_train_config.yaml 
