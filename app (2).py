import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_classification_model():
    model = load_model('model_pneumonia_detection.h5')
    return model

model = load_classification_model()

st.title("PNEUMONIA DETECTION USING CNN")

file = st.file_uploader(
    "Please upload a Chest XRAY IMAGE",
    type=["jpeg", "jpg", "png"]
)

def import_and_predict(image_data, model):
    size = (224, 224)

    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)

    img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)

    st.image(image, use_container_width=True)

    predictions = import_and_predict(image, model)

    class_names = ['Normal', 'Pneumonia']

    result = class_names[np.argmax(predictions)]

    st.success(f"This particular image most likely is: {result}")
