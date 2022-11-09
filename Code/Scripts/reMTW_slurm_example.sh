#!/bin/bash
#SBATCH --job-name=mfinverse
#SBATCH --output=/m/nbe/scratch/megci/MFinverse/reMTW/Data/slurm_out/20subj_%A_%a.out
#SBATCH --error=/m/nbe/scratch/megci/MFinverse/reMTW/Data/slurm_out/20subj_%A_%a_error.out
#SBATCH --open-mode=append
#SBATCH --array=0-23                        # Array numbers are used to select stimuli
#SBATCH --time=04:00:00                     # Takes 3-3.5 h on Tesla V100 GPU
#SBATCH --mem=16G                           # RAM, 16G is more than enough
#SBATCH -c 8                                # Number of CPU cores, GPU is the bottleneck
#SBATCH --gres=gpu:1                        # Request GPU node to save 10x time
#SBATCH --mail-type=END,ARRAY_TASKS         # Email when each array task finishes
#SBATCH --mail-user=paavo.hietala@aalto.fi  # Email receiver

# Array of stimulus names to analyze
stimuli=(sector1 sector2 sector3 sector4 sector5 sector6
         sector7 sector8 sector9 sector10 sector11 sector12
         sector13 sector14 sector15 sector16 sector17 sector18
         sector19 sector20 sector21 sector22 sector23 sector24)

# Pre-computed alpha parameters for re-running parts of the source estimation
alphas=(10.625 12.5 3.0 10.0 10.625 5.625 3.0 3.0 8.125 10.0 3.0 12.5 6.25 3.0
        1.75 2.8 2.5 2.5 4.0 3.0 2.0625 1.6875 0.7 1.1875)

# Pick parameters for this array task run
ALPHA=${alphas[SLURM_ARRAY_TASK_ID]}
STIM=${stimuli[SLURM_ARRAY_TASK_ID]}

# Run unbuffered with -u to update the .out files instantly when something
# happens. You can find a complete list of CLI paramters in README.md.
# For a fresh run remove the -alpha parameter.
srun xvfb-run python -u /m/nbe/scratch/megci/MFinverse/Code/pipeline_reMTW.py \
    -stim=$STIM -target=2 -dir=/m/nbe/scratch/megci/MFinverse/reMTW/ \
    -alpha=$ALPHA
