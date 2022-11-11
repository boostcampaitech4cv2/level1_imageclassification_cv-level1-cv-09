import torch.nn as nn
import torch.nn.functional as F
import torchvision
from main.utils import ArcMarginProduct

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

class EfficientNetEncoderHead(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetEncoderHead, self).__init__()
        self.base = torchvision.models.efficientnet_b2(pretrained = True)
        self.feature_extract = self.base.features
        self.output_filter = self.base.classifier[1].in_features
        self.fc = nn.Linear(self.output_filter, num_classes)
        self.arcface = ArcMarginProduct(in_feature = self.output_filter, out_feature = num_classes)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(self.output_filter)
        

    def forward(self, x, label, training):
        # extract features 
        features = self.extract_features(x) 
        arcface = self.arcface(features, training, label)
        bn = self.bn(features) 
        probs = self.fc(bn) 

        return probs, arcface 

    def extract_features(self, x): 
        batch_size = x.shape[0]
        x = self.feature_extract(x)
        x = self.pooling(x).view(batch_size, -1)
        return x