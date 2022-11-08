#!/bin/sh
SM_CHANNEL_EVAL="/opt/ml/input/data/eval" SM_CHANNEL_MODEL="saved_models/exp55" SM_OUTPUT_DATA_DIR="inference_output"  \
python inference.py --model EfficientB4 \
--single True \
--tta True
# --age_dir "saved_models/joint_exp/age49" \
# --gender_dir "saved_models/joint_exp/gender22" \
# --mask_dir "saved_models/joint_exp/mask23"

