#!/bin/bash
#SBATCH --job-name=mfinverse
#SBATCH --output=/m/nbe/scratch/megci/MFinverse/reMTW/Data/slurm_out/%A_%a.out
#SBATCH --open-mode=append
#SBATCH --array=0-3
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH --gres=gpu:1

case $SLURM_ARRAY_TASK_ID in

    0)  STIM=sector19  ;;
    1)  STIM=sector20  ;;
    2)  STIM=sector21  ;;
    3)  STIM=sector24  ;;

esac

srun xvfb-run python /m/nbe/scratch/megci/MFinverse/Code/pipeline_reMTW.py -stim=$STIM