import torch.nn as nn
import torch.nn.functional as F
import torchvision


class EfficientB4(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientB4, self).__init__()
        self.effnet = torchvision.models.efficientnet_b4(pretrained = True)
        in_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x