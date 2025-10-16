#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=02:30:00
#SBATCH -o /home/akalais/medseg/repo/medical-vit-segmentation/logs/slurm_logs/%x_%j.out


CONTAINER="/home/akalais/medseg/containers/container.sif"
export MEDSEG_DATA_ROOT="/home/akalais/medseg/data"
export MEDSEG_EXPERIMENTS_ROOT="/home/akalais/medseg/repo/medical-vit-segmentation/experiments"

echo "[INFO] DATA_ROOT=$MEDSEG_DATA_ROOT"
echo "[INFO] EXPS_ROOT=$MEDSEG_EXPERIMENTS_ROOT"

srun apptainer exec --nv \
  --env-file .env \
  --bind "$PWD":"$PWD" \
  --bind "$MEDSEG_DATA_ROOT":"$MEDSEG_DATA_ROOT" \
  --bind "$MEDSEG_EXPERIMENTS_ROOT":"$MEDSEG_EXPERIMENTS_ROOT" \
  --pwd "$PWD" \
  "$CONTAINER" /bin/bash main.sh