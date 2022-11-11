import torch
import io
import numpy as np
from PIL import Image
from main.dataset import BaseAugmentation
import torchvision.transforms as transforms
import streamlit as st
from torchvision.transforms.functional import to_pil_image

def transform_image(image_bytes: bytes) -> torch.Tensor:
    transform = BaseAugmentation()
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image_array = np.array(image)
    converted = transform(image=image_array)

    back_to_image = inverse_normalization(converted)
    back_to_image = to_pil_image(back_to_image)
    st.image(back_to_image, caption = "After MTCNN")
    return converted.unsqueeze(0)



def inverse_normalization(tensor) -> torch.Tensor:
    inv_normalize = transforms.Normalize(
    mean=[-0.2063 / 0.3497,  -0.1772 / 0.3085, -0.1677 / 0.2971],
    std=[1/0.3497, 1/0.3085, 1/0.2971])

    inv_tensor = inv_normalize(tensor)
    return inv_tensor