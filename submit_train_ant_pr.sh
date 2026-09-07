#!/bin/bash
#SBATCH --job-name=train_ant_pr
#SBATCH --partition=workq
#SBATCH --account=brics.u6ti
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=23:30:30
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --output=/projects/u6ti/slurm_logs/%j.out
#SBATCH --error=/projects/u6ti/slurm_logs/%j.err

DIFF_DOWNSCALER=/projects/u6ti/diff_downscaler
WORKDIR=/projects/u6ti/diff_downscaler/model_weights/ant_historical_pr

cd ${DIFF_DOWNSCALER}

export PATH="$HOME/.pixi/bin:$PATH"
export DATA_PATH=/projects/u6ti
export EXPERIMENT_NAME=ml-downscaling-emulator-ant-pr

srun pixi run python bin/main.py \
    --config src/ml_downscaling_emulator/configs/subvpsde/ant_historical_pr_cncsnpp_continuous.py \
    --config.training.batch_size=16 \
    --workdir ${WORKDIR} \
    --mode train
