import streamlit as st 
import yaml 
from predict import load_model, get_prediction
import io 
from PIL import Image 

from confirm_button_hack import cache_on_button_press

st.set_page_config(layout = "wide")

def main():
    st.title("Mask Classification model")

    with open("config.yaml") as f:
        config = yaml.load(f, Loader = yaml.FullLoader)

    model = load_model()
    model.eval()

    uploaded_file = st.file_uploader("Choose an image", type = ["jpg", "jpeg", "png"])

    if uploaded_file is not None: 
        #TODO: 이미지 뷰어
        image_bytes = uploaded_file.getvalue()

        #BytesIO를 통해 PIL의 입력으로 넣어줄 수 있다.
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption = "Uploaded Image")
        st.write("Classifying")
        _, y_hat = get_prediction(model, image_bytes)
        label = config["classes"][y_hat.item()]

        st.write(f"Prediction Response is {label}")

#TODO: 보안 

root_password = "doridori"

password = st.text_input("password", type = "password")

#버튼을 누를 때 ....! 한번 코드 읽어보기
@cache_on_button_press("Authenticate")
def authenticate(pasword) -> bool:
    st.write(type(password))
    return password == root_password

if authenticate(password):
    st.success("You are authenticated")
    main()

else: 
    st.error("Password is invalid")



