#!/bin/bash
#SBATCH --job-name=mfinverse
#SBATCH --output=/m/nbe/scratch/megci/MFinverse/Classic/Data/slurm_out/%A_%a.out
#SBATCH --error=/m/nbe/scratch/megci/MFinverse/Classic/Data/slurm_out/%A_%a_error.out
#SBATCH --open-mode=append
#SBATCH --array=0
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH -c 8

# Run unbuffered to update the .out files instantly when something happens
srun xvfb-run python -u /m/nbe/scratch/megci/MFinverse/Code/pipeline_classic.py

#-alpha=$ALPHA -beta=0.5