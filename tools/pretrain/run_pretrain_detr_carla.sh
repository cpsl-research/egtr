#!/usr/bin/env bash

set -e

# Input arguments
DATA_PATH=${1:-"/data/shared/CARLA/scenes-for-egtr/processed"}
OUTPUT_PATH=${2:-"/data/shared/models/detr/model"}
BACKBONE_DIRPATH=${3:-"/data/shared/models/detr/backbone"}
PRETRAIN_DIRPATH=${4:-"/data/shared/models/detr/model/pretrained_detr__SenseTime__deformable-detr/batch__32__epochs__150_50__lr__1e-05_0.0001__visual_genome__finetune/version_0"}
MEMO=${5:-"carla_detection"}
GPUS=${6:-2}

# Finetune DETR on CARLA
python pretrain_detr.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --backbone_dirpath $BACKBONE_DIRPATH \
    --memo $MEMO \
    --max_epochs 150 \
    --max_epochs_finetune 50 \
    --gpus $GPUS \
    --skip_train false \
    --finetune true \
    --load_initial_ckpt false \
    --resume false
    # --initial_ckpt_dir $PRETRAIN_DIRPATH \
