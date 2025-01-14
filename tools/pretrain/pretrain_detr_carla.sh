#!/usr/bin/env bash

set -e

# Input arguments
DATA_PATH=${1:-"/data/shared/CARLA/scenes-for-egtr/processed"}
OUTPUT_PATH=${2:-"pretrain/detr"}
BACKBONE_DIRPATH=${3:-"pretrain/backbone"}
MEMO=${4:-"detr"}

# Pre-train DETR
python pretrain_detr.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --backbone_dirpath $BACKBONE_DIRPATH \
    --memo $MEMO \
    --resume True
