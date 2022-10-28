#!/bin/sh
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --dataset AgeOnlyDataset  --name "age" --criterion "label_smoothing" --lr 0.01
#SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --dataset GenderOnlyDataset  --name "gender" --criterion "label_smoothing"
# SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --dataset MaskOnlyDataset  --name "mask" --criterion "label_smoothing"