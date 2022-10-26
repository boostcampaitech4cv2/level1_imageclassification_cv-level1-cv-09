import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import os 

path = "C:\\Users\\YJ\\Desktop\\project1\\detect_error\\" #이미지 파일이 있는 폴더 경로
for i, pic in enumerate(os.listdir(path)):
    pic_path = path+pic
    img = Image.open(pic_path)

    transforms = nn.Sequential(
        T.RandomAdjustSharpness(sharpness_factor=10, p=1.0)
    )
    sharp_pic = transforms(img)
    sharp_pic.save(pic_path.replace("error", "error_sharp"))

print("done")
