import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

import numpy as np
import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split, WeightedRandomSampler


#Augmentation을 위한 torchvision import
import torchvision
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, RandomErasing, RandomHorizontalFlip

#utils.py에서 rand_bbox, cutmix 함수를 가져옴
from utils import rand_bbox, cutmix


#facenet_pytorch, cv2(opencv)는 pip을 통해 별도로 설치가 필요함
#하단 주석 참고
from facenet_pytorch import MTCNN

IMG_EXTENSIONS = [
    ".jpg", ".JPG", ".jpeg", ".JPEG", ".png",
    ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
]


"""
SM : 이미지 확장자인지 확인하는 함수. any(  )를 통해 하나라도 확장자가 맞는지 확인한다.
"""
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
 
"""Args
resize    : ImageNet pretrained model이 학습된 기본 사이즈로 resize
mean, std : ImageNet pretrained model에 쓰인 기본값

이미지넷 데이터의 mean,std가 안리ㅏ 우리의 데이터의 mean,std ===> 그렇게 실험을 한 기록이 있었습니다 지난 기수 때도 한번 해보세요! 
"""
class BaseAugmentation:
    def __init__(self, resize=[380, 380], mean=(0.2063, 0.1772, 0.1677), std=(0.3497, 0.3085, 0.2971)):
        self.transform = Compose([
            ToTensor(),
            FaceNet(size=resize),          #FaceNet : CenterCrop과 비슷한 역할을 하는 Augmentation 라이브러리
            Normalize(mean=mean, std=std)							   
        ])

    def __call__(self, image):
        return self.transform(image)


"""
FaceNet

- https://yeomko.tistory.com/16
- Implementation : Please refer [facenet_pytorch](https://github.com/timesler/facenet-pytorch)
- Or, visit "https://www.kaggle.com/code/timesler/guide-to-mtcnn-in-facenet-pytorch/notebook"

tensor<=> PIL 간에 상호변환하는 함수
detect하는 함수를 만든다.

EDA결과 대부분의 사진이 정방향을 향하도록 되어있지만,

detect 결과가 None이면 CenterCrop, 있으면 그걸 그대로 이용.
"""
class FaceNet(object):
    def __init__(self, size):
        self.tensor_to_PIL = torchvision.transforms.ToPILImage()
        self.PIL_to_tensor = torchvision.transforms.ToTensor()
        self.center_crop = torchvision.transforms.CenterCrop(size)


        self.face_detector = MTCNN(image_size=size[0], margin=150, post_process=False)

    def __call__(self, tensor):
        img = self.tensor_to_PIL(tensor)
        face = self.face_detector(img) # Tensor type

        if face == None:
            face = self.center_crop(tensor) # Tensor type
        else:
            face = PIL.ImageOps.invert(self.tensor_to_PIL(face))  #PIL type
            face = self.PIL_to_tensor(face)  # Back to Tensor type
        return face


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = Compose([					   							 
            ToTensor(),
            FaceNet(size=resize),
            RandomErasing(p=1, scale=(0.05,0.05), ratio=(0.5,1)),
            Normalize(mean=mean, std=std)
        ])

    def __call__(self, image):
        return self.transform(image)


class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 59:
            return cls.MIDDLE
        else:
            return cls.OLD


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3

    _file_names = {
        "mask1": MaskLabels.MASK,
        "mask2": MaskLabels.MASK,
        "mask3": MaskLabels.MASK,
        "mask4": MaskLabels.MASK,
        "mask5": MaskLabels.MASK,
        "incorrect_mask": MaskLabels.INCORRECT,
        "normal": MaskLabels.NORMAL
    }


    def __init__(self, data_dir, mean=(0.2063, 0.1772, 0.1677), std=(0.3497, 0.3085, 0.2971), val_ratio=0.2, age_removal = False):
        self.data_dir = data_dir
        self.mean = mean
        self.std = std
        self.val_ratio = val_ratio
        self.age_removal = age_removal
        
        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        self.class_labels = []



        self.transform = None
        self.setup()
        self.calc_statistics()

    #설명 : setup 함수는 data_dir의 파일을 읽어, metadata를 읽어준다. X는 이미지 경로, y는 각각의 라벨(이 라벨은, 위에서 정의한 Enum 클래스를 이용)
    def setup(self, age_remove = False):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")

                #추가 :: Age removal
                """
                경계선에 있는 값들은 예측하기 어렵다.

                경우에 따라, 경계선에 있는 데이터들을 제거해볼 수 있다.
                """
                if self.age_removal:
                    if (27<=int(age)<=29) or (57<=int(age)<=59) :
                        continue
                #추가 끝

                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)

    """
    설명 : mean,std가 있는지 확인하여, 없으면 이를 각각 계산해줌 => 이는 이후의 Normalize를 위함.
    """
    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform


    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)


        return self.transform(image), multi_class_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def read_image(self, index):
        image_path = self.image_paths[index]
        return Image.open(image_path)

    @staticmethod
    def encode_multi_class(mask_label, gender_label, age_label) -> int:
        return mask_label * 6 + gender_label * 3 + age_label

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self) -> Tuple[Subset, Subset]:
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = random_split(self, [n_train, n_val])
        return train_set, val_set


class TestDataset(Dataset):
    def __init__(self, img_paths, resize , mean=(0.2063, 0.1772, 0.1677), std=(0.3497, 0.3085, 0.2971), tta = False):
        self.img_paths = img_paths
        self.tta = tta

        
        if not self.tta:
            self.transform = Compose([
                Resize(resize, Image.BILINEAR),
                ToTensor(),
                Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = BaseAugmentation()

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)


class MaskSplitByProfileDataset(MaskBaseDataset):
    """
        train / val 나누는 기준을 이미지에 대해서 random 이 아닌
        사람(profile)을 기준으로 나눕니다.
        구현은 val_ratio 에 맞게 train / val 나누는 것을 이미지 전체가 아닌 사람(profile)에 대해서 진행하여 indexing 을 합니다
        이후 `split_dataset` 에서 index 에 맞게 Subset 으로 dataset 을 분기합니다.
    """

    def __init__(self, data_dir, mean=(0.2063, 0.1772, 0.1677), std=(0.3497, 0.3085, 0.2971), val_ratio=0.2, age_removal = False):
        self.indices = defaultdict(list)
        super().__init__(data_dir, mean, std, val_ratio, age_removal)


    """
    profiles 을 train_indices, val_indices로 쪼갭니다.
    """

    @staticmethod
    def _split_profile(profiles, val_ratio):
        length = len(profiles)
        n_val = int(length * val_ratio)

        val_indices = set(random.sample(range(length), k=n_val))
        train_indices = set(range(length)) - val_indices
        return {
            "train": train_indices,
            "val": val_indices
        }

    """
    위에서 쪼갠 train_indices, val_indices에 대해 각각 append하는 구조입니다. 
    """
    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)
        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]
                    id, gender, race, age = profile.split("_")

                    #추가
                    if self.age_removal:
                        if (27<=int(age)<=29) or (57<=int(age)<=59) :
                            continue
                    #추가 끝

                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    self.indices[phase].append(cnt)
                    cnt += 1

    """
    - Subset(dataset,indices)을 하면 indices만 별도로 가지는 dataset을 지정할 수 있습니다.
    """

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]

    """
    __getitem__ 함수 추가
    """
    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        if index in self.indices["train"] :
            _transform =  self.transform
        
        #더 정확한 valid 탐지를 위해, valid에도 BaseAugmentation()을 적용!
        else :
            _transform =  BaseAugmentation()
        return _transform(image), multi_class_label

    """
    sampler 함수 추가. thx.to. 한별누나 코드

    - Sampler는, dataset에서 data를 뽑을 때 샘플의 분포를 고려하여 뽑아주는 코드
    - 다만 그러다 보니 생기는 문제는, 희귀한 label을 여러번 뽑게 된다는 점.
    - 하여 적당한 Data Augmentation이 필요하다.
    """

    def get_sampler(self, phase) :
        _multi_class = []
        for phase_idx in self.indices[phase]:
            _temp = self.encode_multi_class(self.mask_labels[phase_idx],
                                    self.gender_labels[phase_idx],
                                    self.age_labels[phase_idx])
            _multi_class.append(_temp)
       
        class_sample_count = np.array([len(np.where(_multi_class == t)[0]) for t in np.unique(_multi_class)])		   
        weight = 1. / class_sample_count
								  
        samples_weight = np.array([weight[t] for t in _multi_class])
        samples_weight = torch.from_numpy(samples_weight).double()
        phase_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return phase_sampler

"""
추가:
CutMixDataset and its variants:

- CutMix를 이용하여 학습을 진행!

찾아보니 CutMix의 구현엔 대략 2가지가 있었다.

[1] Train 단계에서 cutMix를 이용한다.
=> 각 Batch 내에서 random하게 섞는 방법! 

[2] __getitem__ 단계에서 cutmix를 이용한다.

=> 데이터를 가져올때마다 random하게 cutmix를 이용하는 방법
하단의 예시는 [2]의 방식으로 구현된 cutmix

"""
class CutMixDataset(MaskSplitByProfileDataset):
    """
    MaskSplitByProfileDataset에 Cutmix를 적용한다.
    """

    def __init__(self, data_dir, mean=(0.2063, 0.1772, 0.1677), std=(0.3497, 0.3085, 0.2971), val_ratio=0.2, age_removal = False):
        self.indices = defaultdict(list)

        #Added : 각 class_idx 별 indices를 저장하고자 함
        self.class_idx = [[] for i in range(18)]
        self.istrain = []
        super().__init__(data_dir, mean, std, val_ratio, age_removal)
        
        
    def setup(self):
        profiles = os.listdir(self.data_dir)
        profiles = [profile for profile in profiles if not profile.startswith(".")]
        split_profiles = self._split_profile(profiles, self.val_ratio)


        cnt = 0
        for phase, indices in split_profiles.items():
            for _idx in indices:
                profile = profiles[_idx]
                img_folder = os.path.join(self.data_dir, profile)
                for file_name in os.listdir(img_folder):
                    _file_name, ext = os.path.splitext(file_name)
                    if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                        continue

                    img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                    mask_label = self._file_names[_file_name]

                    id, gender, race, age = profile.split("_")
                    if self.age_removal:
                        if (27<=int(age)<=29) or (57<=int(age)<=59) :
                            continue
                    gender_label = GenderLabels.from_str(gender)
                    age_label = AgeLabels.from_number(age)

                    self.image_paths.append(img_path)
                    self.mask_labels.append(mask_label)
                    self.gender_labels.append(gender_label)
                    self.age_labels.append(age_label)

                    multi_class = self.encode_multi_class(mask_label,gender_label,age_label)

                    self.class_idx[multi_class].append(cnt)
                    self.istrain.append(phase)

                    self.indices[phase].append(cnt)
                    cnt += 1


    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"
        _image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        
        multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)

        #Train시킬 이미지는 Val 이미지와 Cutmix되어서는 안되고, 일정 확률로 정의해야한다
        #이걸 바꿀거면, random.rand() > 0.5(0.46).. 등으로 커스텀 가능!

        if self.istrain[index] == "train" and random.choice([True,False]):
            mix_idx = self.get_rand_idx(multi_class_label)
            _mix_image = self.read_image(mix_idx)
            _image = self.transform(_image)
            _mix_image = self.transform(_mix_image)

            image = cutmix(_image, _mix_image)
        else :           
            image = BaseAugmentation()(_image)

        """
        60대 이상에 대하여, 추가적으로 Augmentation을 줄 수 있다!
        """
        if age_label in [2]:
            #60대는 랜덤하게 뒤집어줌
            image = RandomHorizontalFlip()(image)
            
        return image, multi_class_label
    
    #0~ 18 중 랜덤하게 아무 index나 뽑는다.
    def get_rand_idx(self,label):
        return random.choice(self.class_idx[label])

    def split_dataset(self) -> List[Subset]:
        return [Subset(self, indices) for phase, indices in self.indices.items()]