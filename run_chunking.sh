#!/bin/bash
#SBATCH --job-name=chunking
#SBATCH --account=kempner_mzitnik_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=200G
#SBATCH --output=chunking_output.out
#SBATCH --error=chunking_error.err
#SBATCH --mail-type=END
#SBATCH --mail-user=lavikjain@college.harvard.edu

conda activate /n/holylfs06/LABS/mzitnik_lab/Lab/phyla_data_share/phyla
module load cuda/12.2.0-fasrc01
/n/holylfs06/LABS/mzitnik_lab/Lab/phyla_data_share/phyla/bin/python3 chunking.py