#!/bin/bash
#SBATCH --job-name=mfinverse
#SBATCH --output=/m/nbe/scratch/megci/MFinverse/reMTW/Data/slurm_out/20subj_%A_%a.out
#SBATCH --error=/m/nbe/scratch/megci/MFinverse/reMTW/Data/slurm_out/20subj_%A_%a_error.out
#SBATCH --open-mode=append
#SBATCH --array=0,3,19
#SBATCH --time=03:30:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH --gres=gpu:1

stimuli=(sector1 sector2 sector3 sector4 sector5 sector6
         sector7 sector8 sector9 sector10 sector11 sector12
         sector13 sector14 sector15 sector16 sector17 sector18
         sector19 sector20 sector21 sector22 sector23 sector24)

alphas=(10.625 12.5 3.0 10.0 10.625 5.625 3.0 3.0 8.125 10.0 3.0 12.5 6.25 3.0
        1.75 2.8 2.5 2.5 4.0 3.0 2.0625 1.6875 0.7 1.1875)

ALPHA=${alphas[SLURM_ARRAY_TASK_ID]}
STIM=${stimuli[SLURM_ARRAY_TASK_ID]}

# Run unbuffered to update the .out files instantly when something happens
srun xvfb-run python -u /m/nbe/scratch/megci/MFinverse/Code/pipeline_reMTW.py \
    -stim=$STIM -target=2 -dir=/m/nbe/scratch/megci/MFinverse/reMTW/

#-alpha=$ALPHA -beta=0.5