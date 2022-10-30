#!/bin/sh
# SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --dataset AgeProfileOnlyDataset  --name "age" --cutmix "yes"
# SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --dataset GenderProfileOnlyDataset  --name "gender" --cutmix "yes"
# SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models/joint_exp" python train.py  --dataset MaskProfileOnlyDataset  --name "mask" --cutmix "yes"

#SINGLE
#SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models" python train.py  --dataset MaskProfileOnlyDataset  --name "exp" --cutmix "yes"
SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models" python train.py  --dataset MaskSplitByProfileDataset  --name "exp" --model "EfficientB4" --cutmix "yes"
#SM_CHANNEL_TRAIN="/opt/ml/input/data/train/images" SM_MODEL_DIR="saved_models" python train.py  --dataset MaskBaseDataset  --name "exp" 
