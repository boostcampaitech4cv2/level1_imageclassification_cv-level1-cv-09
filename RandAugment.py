from torch.utils.data import *
from torchvision.transforms import *

class Custom_equalize() :
    def __init__(self) :
        self.transforms = Compose([
            RandomEqualize( p = 0.3 )
        ])

    def __call__(self, image) :
        return self.transforms(image)

class Custom_solarize() :
    def __init__(self) :
        self.transforms = Compose([
            RandomSolarize( 0, p=0.3 )
        ])
    
    def __call__(self, image) :
        return self.transforms(image)

# class Custom_color() :
#     def __init__(self) :
#         self.transforms = Compose([
#             ColorJitter(
#                 brightness=0.1,
#                 contrast=0.1,
#                 saturation=0.1,
#                 hue=0.1
#             )
#         ])

#     def __call__(self, image) :
#         return self.transforms(image)
