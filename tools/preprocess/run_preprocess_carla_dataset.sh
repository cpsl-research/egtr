#!/usr/bin/env bash

set -e


# Input arguments
INPUT_DIR=${1:-"/data/shared/CARLA/scenes-for-egtr/raw"}
OUTPUT_DIR=${2:-"/data/shared/CARLA/scenes-for-egtr/processed"}
STRIDE=${3:-4}

# Run preprocessing
python preprocess_carla_dataset.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --stride $STRIDE