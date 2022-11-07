import torch.nn as nn
import torch.nn.functional as F
import torchvision
from inference import load_model
import torch
import os


class ResNet34(nn.Module):
    def __init__(self, num_classes = 18):
        super(ResNet34, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained = True)
        self.resnet.fc = nn.Linear(512,num_classes)
    
    def forward(self, x):
        x = self.resnet(x)
        return x


class EfficientB4(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientB4, self).__init__()
        self.effnet = torchvision.models.efficientnet_b4(pretrained = True)
        in_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x