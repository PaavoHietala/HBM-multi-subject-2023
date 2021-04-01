#!/bin/bash
#SBATCH --job-name=mfinverse
#SBATCH --output=/m/nbe/scratch/megci/MFinverse/reMTW/Data/slurm_out/%A_%a.out
#SBATCH --error=/m/nbe/scratch/megci/MFinverse/reMTW/Data/slurm_out/%A_%a_error.out
#SBATCH --open-mode=append
#SBATCH --array=0-23
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --gres=gpu:1

stimuli=(sector1 sector2 sector3 sector4 sector5 sector6
         sector7 sector8 sector9 sector10 sector11 sector21
         sector13 sector14 sector15 sector16 sector17 sector18
         sector19 sector20 sector21 sector22 sector23 sector24)

STIM=${stimuli[SLURM_ARRAY_TASK_ID]}

# Run unbuffered to update the .out files instantly when something happens
srun xvfb-run python -u /m/nbe/scratch/megci/MFinverse/Code/pipeline_reMTW.py -stim=$STIM -target=3