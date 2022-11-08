import numpy as np
import matplotlib.pyplot as plt
import torch
import os 
import shutil
from torch.nn import Parameter
import torch.nn.functional as F
import math
import torch.nn as nn 


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
https://sseunghyuns.github.io/classification/2021/05/25/invasive-pytorch/#
원래는 Beta분포로 랜덤하게 Bbox를 합쳐야 하지만,
학습의 용이성을 위해 절반만 잘라붙었음.
"""

def rand_bbox(size, lam):
    W = size[-2]  # C x W x H  (3D) or B x C x W x H (4D)
    H = size[-1]

    bbx1 = np.clip(0,0,W)
    bby1 = np.clip(H // 2,0,H)
    bbx2 = np.clip(W,0,W)
    bby2 = np.clip(H,0,H)

    return bbx1, bby1, bbx2, bby2

def cutmix(im1,im2):
    #print(im1)
    bbx1, bby1, bbx2, bby2 = rand_bbox(size = im2.size(), lam = 0) #동일 라벨일 경우 ram 생략
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


## arcface loss ##
class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=1000, out_feature=18, s=64.0, m=0.2, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.in_feature = in_feature
        self.out_feature = out_feature # 클래스 개수 
        self.s = s # feature scale; 나중엔 나온 logits를 s로 곱해줌 
        self.m = m # margin 

		# 입력차원 x 클래스 개수로 이루어진 매트릭스 
        self.weight = Parameter(torch.Tensor(out_feature, in_feature)).to(self.device)
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, training, label=None):
        # cos(theta): 입력이 속한 class의 중심과 입력의 각도 
		# input과 weight를 noramlize해서 길이 1인 구 위에 위치할 수 있게 함
		# 내적해서 값이 크면 class 중심과 입력의 각도가 작다는 의미 (거리가 가깝다)
		# 값이 작으면 각도가 크다 (거리가 멀다) 
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))

        if training: 
            # cos(theta + m) = cos(theta)*cos(m) + sin(theta)*sin(m) 
            # margin m을 추가; 각도를 더 크게 계산하는 효과 
            # 해당 class에 속할 가능성이 실제에 비해 더 낮게 계산되도록 함 
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m

            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
            
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        else: 
            output = cosine 

        output = output * self.s

        return output

def cal_loss(criterion, model, inputs, labels):
    if criterion == 'arcface': 
        outs, af = model(inputs, labels, training=True) 
        preds = None 

        loss = nn.CrossEntropyLoss()(af, labels) 
        output = nn.Softmax()(outs)
        _, preds = torch.topk(output, 1)
        preds = preds.squeeze() 

    else: 
        outs = model(inputs)
        preds = torch.argmax(outs, dim=-1)
        loss = None
    
    return outs, preds, loss 