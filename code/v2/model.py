import torch.nn as nn
import torch.nn.functional as F
import torchvision
from inference import load_model

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes = 18):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained = True)
        self.resnet.fc = nn.Linear(512,num_classes)
    
    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, num_classes = 18):
        super(ResNet34, self).__init__()
        self.resnet = torchvision.models.resnet34(pretrained = True)
        self.resnet.fc = nn.Linear(512,num_classes)
    
    def forward(self, x):
        x = self.resnet(x)
        return x

class ResNet101(nn.Module):
    def __init__(self, num_classes = 18):
        super(ResNet101, self).__init__()
        self.resnet = torchvision.models.resnet101(pretrained = True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features , num_classes)
    
    def forward(self, x):
        x = self.resnet(x)
        return x

class EfficientV2(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientV2, self).__init__()
        self.effnet = torchvision.models.efficientnet_v2_s(pretrained = True)
        in_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x

class Efficient(nn.Module):
    def __init__(self, num_classes = 18):
        super(Efficient, self).__init__()
        self.effnet = torchvision.models.efficientnet_b2(pretrained = True)
        in_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
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

class Vit_B_16(nn.Module):
    def __init__(self, num_classes = 18):
        super(Vit_B_16,self).__init__()
        self.vit = torchvision.models.vit_b_16(pretrained = True)
        self.vit.heads = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(256,num_classes)
            )
        for param, weight in self.vit.named_parameters():
            if param.split('.')[0] == 'heads': pass
            else : weight.requires_grad = True

    def forward(self,x):
        x = self.vit(x)
        return x



class EfficientB4_pretrained(nn.Module):
    def __init__(self, num_classes = 18):
        super(EfficientB4_pretrained, self).__init__()
        self.effnet = torchvision.models.efficientnet_b4(pretrained = True)
        in_features = self.effnet.classifier[1].in_features
        self.effnet.classifier[1] = nn.Linear(in_features , num_classes)

    def forward(self, x):
        x = self.effnet(x)
        return x