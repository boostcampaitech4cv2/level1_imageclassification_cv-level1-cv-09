#!/bin/sh
python train.py --model EfficientB4 \
--criterion f1 \
--optimizer Adam \
--epochs 10 \
--lr 0.0001 \
--dataset MaskSplitByProfileDataset \
--augmentation CustomAugmentation \
--batch_size 32 \
--age_removal True \
--val_ratio 0.2 \
--valid_batch_size 256 \
--val_train True \
--val_epochs 3 \
--cutmix yes \
--cutmix_prob 0.7 \
--cutmix_lower 0.46 \
--cutmix_upper 0.54 \
--seed 7