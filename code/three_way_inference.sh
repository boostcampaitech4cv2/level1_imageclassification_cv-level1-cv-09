#!/bin/sh
SM_CHANNEL_EVAL="/opt/ml/input/data/eval" SM_CHANNEL_MODEL="saved_models/exp6" SM_OUTPUT_DATA_DIR="inference_output"  \
python inference.py --model EfficientV2 \
--single False \
--age_dir "saved_models/joint_exp/age31" \
--gender_dir "saved_models/joint_exp/gender18" \
--mask_dir "saved_models/joint_exp/mask16"