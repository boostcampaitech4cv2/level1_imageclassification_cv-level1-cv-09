import argparse 
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import os 
import shutil

DATA_CLASS_MODULE = "dataset"
MODEL_CLASS_MODULE = "models"
TRAIN_DATA_DIR = "../input/data/train/images"


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

"""
Original cutmix
논문 구현 : https://github.com/clovaai/CutMix-PyTorch/blob/2d8eb68faff7fe4962776ad51d175c3b01a25734/train.py#L279
https://sseunghyuns.github.io/classification/2021/05/25/invasive-pytorch/#
원래는 Beta분포로 랜덤하게 Bbox를 합쳐야 하지만,
학습의 용이성을 위해 절반만 잘라붙었음.
"""

def rand_bbox(size, lam):
    W = size[-2]  # C x W x H  (3D) or B x C x W x H (4D)
    H = size[-1]

    bbx1 = np.clip(0,0,W)
    bby1 = np.clip(int(lam*H),0,H)
    bbx2 = np.clip(W,0,W)
    bby2 = np.clip(H,0,H)

    return bbx1, bby1, bbx2, bby2

def cutmix(im1,im2, lam = 0.5):

    bbx1, bby1, bbx2, bby2 = rand_bbox(size = im2.size(), lam = lam) #동일 라벨일 경우 ram 생략
    im1[:, bbx1:bbx2, bby1:bby2] = im2[:, bbx1:bbx2, bby1:bby2]
    return im1


def cutmix_plot(train_loader):
    fig , axes = plt.subplots(1,3)
    fig.set_size_inches(15,12)
    
    for i in range(3):
        for inputs, targets in train_loader:
            inputs = inputs
            targets = targets
            break

        lam = np.random.beta(1.0, 1.0) 
        rand_index = torch.randperm(inputs.size()[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
        axes[i].imshow(inputs[1].permute(1, 2, 0).cpu())
        axes[i].set_title(f'λ : {np.round(lam,3)}')
        axes[i].axis('off')
        plt.savefig(f"tmp_{i}.png", dpi = 200)
    return


def remove_strange_files():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    for path, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(path, file)
            if file.startswith('._'):
                os.remove(file_path)
                print(f'removed {file_path}')


def copy_files_to_upper_dir():
    dir_path = os.path.dirname("/opt/ml/input/data/train/images/")
    for subdir in os.listdir(dir_path):
        filepath = os.path.join(dir_path, subdir)
        if subdir.endswith("jpg") or subdir.endswith("jpeg"): continue
        for file in os.listdir(filepath):
            old_path = os.path.join(filepath, file)
            new_name = subdir + "_"+ file
            new_path = os.path.join(dir_path, new_name)
            shutil.copy(old_path, new_path)