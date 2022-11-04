#!/bin/sh
SM_CHANNEL_EVAL="/opt/ml/input/data/eval" SM_CHANNEL_MODEL="model/exp53" SM_OUTPUT_DATA_DIR="inference_output"  \
python inference.py --model EfficientB4 \
--single True \
--tta True