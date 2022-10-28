import argparse 
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch

DATA_CLASS_MODULE = "dataset"
MODEL_CLASS_MODULE = "models"
TRAIN_DATA_DIR = "../input/data/train/images"

def import_class(module_and_class_name):
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def setup_data_and_model_from_args(args: argparse.Namespace, data_class_module = DATA_CLASS_MODULE, model_class_module = MODEL_CLASS_MODULE):
    data_class = import_class(f"{data_class_module}.{args.data_class}")
    model_class = import_class(f"{model_class_module}.{args.model_class}")

    dataset = data_class(TRAIN_DATA_DIR)

    #Fine-tuning 결과에 따라 달라질 것임
    model = model_class(dataset.num_classes)
    return dataset, model

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

"""
https://sseunghyuns.github.io/classification/2021/05/25/invasive-pytorch/#
"""

def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(im1,im2):

    bbx1, bby1, bbx2, bby2 = rand_bbox(size = im2.size(), lam = 0) 
    im1[:, bbx1:bbx2, bby1:bby2] = im2[:, bbx1:bbx2,bby1:bby2]
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