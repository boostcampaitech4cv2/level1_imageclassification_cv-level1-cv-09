from dataset import MaskSplitByProfileDataset, BaseAugmentation
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
data_dir = "/opt/ml/input/data/train/images"
data = MaskSplitByProfileDataset(data_dir = data_dir)
data.get_sampler("train")
data.set_transform(BaseAugmentation(resize = (224,224), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)))

#Loader 정의 train sampler 
train_loader = DataLoader(
    data,
    64, # batch size
    num_workers=0,
    shuffle=False,
    drop_last=True,
    sampler = data.get_sampler("train") # WeightedRandomSampler
    )
label_count = [0 for i in range(0,18)]
for _, label in tqdm(train_loader):
    label = np.array(label)
    for i in label:
        label_count[i] += 1

print(label_count)