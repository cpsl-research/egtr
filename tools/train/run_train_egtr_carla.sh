#!/usr/bin/env bash

set -e

# Input arguments
DATA_PATH=${1:-"/data/shared/CARLA/scenes-for-egtr/processed"}
OUTPUT_PATH=${2:-"/data/shared/models/egtrs/model"}
BACKBONE_DIRPATH=${3:-"/data/shared/models/detr/backbone"}
MEMO=${4:-"carla_sgg"}
GPUS=${5:-2}


# Train EGTR on CARLA
python train_egtr.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --backbone_dirpath $BACKBONE_DIRPATH \
    --memo $MEMO \
    --from_scratch true \
    --pretrained architecture \
    --gpus 2 \
    --skip_train false \
    --finetune true \
    --resume true

        # --gpus $GPUS \
