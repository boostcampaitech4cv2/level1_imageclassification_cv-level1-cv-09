#!/bin/sh
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --data_class AgeOnlyDataset  --name "age"
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --data_class GenderOnlyDataset  --name "gender"
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --data_class MaskOnlyDataset  --name "mask"