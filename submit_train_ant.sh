#!/bin/bash
#SBATCH --job-name=train_ant_mv
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --qos=orchid
#SBATCH --gres=gpu:1
#SBATCH --time=23:30:30
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16 
#SBATCH --output=/gws/ssde/j25a/extant/users/ssharma/data_processed_addison/logs/%j.out
#SBATCH --error=/gws/ssde/j25a/extant/users/ssharma/data_processed_addison/logs/%j.err

DIFF_DOWNSCALER=/gws/ssde/j25a/extant/users/ssharma/diff_downscaler
WORKDIR=/gws/ssde/j25a/extant/users/ssharma/data_processed_addison/model_weights/ant_historical_mv

cd ${DIFF_DOWNSCALER}

pixi run python bin/main.py \
    --config src/ml_downscaling_emulator/configs/subvpsde/ant_historical_mv_cncsnpp_continuous.py \
    --workdir ${WORKDIR} \
    --mode train
