#!/usr/bin/env bash

set -e

# Input arguments
DATA_PATH=${1:-"/data/shared/CARLA/scenes-for-egtr/processed"}
OUTPUT_PATH=${2:-"/data/shared/models/egtr/model"}
BACKBONE_DIRPATH=${3:-"/data/shared/models/detr/backbone"}
DETR_DIRPATH=${4:-'/data/shared/models/detr/model/pretrained_detr__SenseTime__deformable-detr/batch__8__epochs__150_50__lr__1e-05_0.0001__carla_detection__finetune/version_0/'}
MEMO=${4:-"carla_sgg"}
GPUS=${5:-2}


# Train EGTR on CARLA
CUDA_LAUNCH_BLOCKING=1
python train_egtr.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --backbone_dirpath $BACKBONE_DIRPATH \
    --memo $MEMO \
    --from_scratch false \
    --pretrained $DETR_DIRPATH \
    --gpus 2 \
    --skip_train false \
    --finetune true \
    --max_epochs 100 \
    --max_epochs_finetune 40 \
    --resume true

        # --gpus $GPUS \
















# set -e

# # Input arguments
# DATA_PATH=${1:-"/data/shared/CARLA/scenes-for-egtr/processed"}
# OUTPUT_PATH=${2:-"/data/shared/models/egtrs/model"}
# BACKBONE_DIRPATH=${3:-"/data/shared/models/detr/backbone"}
# MEMO=${4:-"carla_sgg"}
# GPUS=${5:-2}


# # Train EGTR on CARLA
# python train_egtr.py \
#     --data_path $DATA_PATH \
#     --output_path $OUTPUT_PATH \
#     --backbone_dirpath $BACKBONE_DIRPATH \
#     --memo $MEMO \
#     --from_scratch true \
#     --pretrained architecture \
#     --gpus 1 \
#     --max_epochs 50 \
#     --max_epochs_finetune 50 \
#     --skip_train false \
#     --finetune true \
#     --resume true

#         # --gpus $GPUS \




