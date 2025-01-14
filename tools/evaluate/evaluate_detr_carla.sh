#!/usr/bin/env bash

set -e

# Input arguments
DATA_PATH=${1:-"/data/shared/CARLA/scenes-for-egtr/processed"}
OUTPUT_PATH=${2:-"pretrain/detr/pretrained_detr__SenseTime__deformable-detr/batch__8__epochs__150_50__lr__1e-05_0.0001__detr/version_0"}

# Evaluate just the detr capability
python evaluate_egtr.py \
    --data_path $DATA_PATH \
    --artifact_path $OUTPUT_PATH \
    --split "test" \
    --eval_single_preds False \
    --eval_multiple_preds False \
    --detr_only