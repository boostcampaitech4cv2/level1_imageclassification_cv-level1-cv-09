#!/bin/sh
python train.py --model EfficientB4 \
--criterion cross_entropy \
--optimizer Adam \
--epochs 4 \
--lr 0.0001 \
--dataset MaskSplitByProfileDataset \
--augmentation CustomAugmentation \
--batch_size 32 \
--age_removal True \
--val_ratio 0.2 \
--valid_batch_size 256 \
--val_train true \
--val_epochs 1 \
--cutmix yes \
--cutmix_prob 0.4 \
--cutmix_lower 0.46 \
--cutmix_upper 0.54