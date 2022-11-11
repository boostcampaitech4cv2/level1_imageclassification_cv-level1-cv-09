import torch 
import streamlit as st 
from main.model import EfficientB4
from app_utils import transform_image 
import yaml 
from typing import Tuple 

@st.cache 
def load_model() -> EfficientB4:
    with open("config.yaml") as f:
        config = yaml.load(f, Loader = yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientB4(num_classes = 18).to(device)
    print("path:", config["model_path"])
    model.load_state_dict(torch.load(config["model_path"], map_location = device))

    return model 

def get_prediction(model : EfficientB4 , image_bytes : bytes) -> Tuple[torch.Tensor, torch.Tensor]:
    st.write("Processing...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessed = transform_image(image_bytes = image_bytes)
    tensor = preprocessed.to(device)
    outputs = model.forward(tensor)
    _, y_hat = outputs.max(1)
    return tensor, y_hat

