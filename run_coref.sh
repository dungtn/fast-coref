#! /bin/bash
#SBATCH --job-name=fast-coref
#SBATCH --output=logs/fast-coref-%A_%a.out
#SBATCH --partition=rtx8000-short
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --array=0-1

module load cuda11/11.2.1
module load cudnn/8.1-cuda_11.2
#conda activate genre_env
conda activate /home/dthai/miniconda3/envs/fast-coref
source /mnt/nfs/scratch1/dthai/fast-coref/wandb_settings.sh
export PYTHONPATH=/mnt/nfs/scratch1/dthai/fast-coref/src/:$PYTHONPATH

# Set to scratch/work since server syncing will occur from here
# Ensure sufficient space else runs crash without error message
export WANDB_DIR="./wandb_dir/"
export TMPDIR="./tmp_dir/"
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES wandb agent dthai/fast-coref/t8air4tf

