#!/bin/bash
#SBATCH --job-name=mfinverse
#SBATCH --output=/m/nbe/scratch/megci/MFinverse/Classic-validation/Data/slurm_out/%A_%a.out
#SBATCH --error=/m/nbe/scratch/megci/MFinverse/Classic-validation/Data/slurm_out/%A_%a_error.out
#SBATCH --open-mode=append
#SBATCH --array=0
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH -c 16
#SBATCH --mail-type=END,ARRAY_TASKS         # Email when each array task finishes
#SBATCH --mail-user=paavo.hietala@aalto.fi  # Email receiver

# Run unbuffered to update the .out files instantly when something happens
srun xvfb-run python -u /m/nbe/scratch/megci/MFinverse/Code/pipeline_classic.py

#-alpha=$ALPHA -beta=0.5