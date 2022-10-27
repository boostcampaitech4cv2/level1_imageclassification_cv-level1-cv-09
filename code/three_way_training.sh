#!/bin/sh
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --data_class AgeOnlyDataset  --name "age" --criterion "label_smoothing"
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --data_class GenderOnlyDataset  --name "gender" --criterion "label_smoothing"
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --data_class MaskOnlyDataset  --name "mask" --criterion "label_smoothing"