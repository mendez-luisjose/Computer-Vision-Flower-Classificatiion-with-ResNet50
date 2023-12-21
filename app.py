import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from skimage.transform import resize
from util import set_background
import json

set_background("./imgs/background.png")

with open("./model/model_archiquecture", "r") as f:
  loaded_json = f.read()

LABELS_PATH = "./labels/cat_to_name.json"

f = open(LABELS_PATH)

labels = json.load(f)

model = keras.models.model_from_json(loaded_json)

model.load_weights("./model/model_weights.h5")

IMG_SIZE_MODEL = [224, 224]

def predict(img):
    img_resize = resize(img, (IMG_SIZE_MODEL[0], IMG_SIZE_MODEL[1]))
    X = tf.keras.applications.imagenet_utils.preprocess_input(img_resize*255)

    X = np.expand_dims(X, axis = 0)

    X = np.vstack([X])

    result = model.predict(X)
    index = np.argmax(result)
    percentage = result[0][index] * 100

    index = str(index +1)

    flower = "Flower: " + labels[index].upper()
    acc = "Accuracy: " + "%.2f" % percentage + "%"

    return flower, acc


header = st.container()
body = st.container()

st.sidebar.markdown('''
    🧑🏻‍💻 Created by [Luis Jose Mendez](https://github.com/mendez-luisjose).
    ''')


st.sidebar.markdown("---------")

st.sidebar.title("Flower Classification Predictor App 🌸")

st.sidebar.image("./imgs/image_00005.jpg", use_column_width=True)

st.sidebar.markdown("---------")

st.markdown(
    """
    <style>
    [data-testid="stSidebar][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }

    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.subheader("Please Submit a Flower Image:")
img_file = st.sidebar.file_uploader("Upload the Image:", type=["jpg", "png", "jpeg"])

if img_file is not None :
    image = np.array(Image.open(img_file))

    st.sidebar.image(image)

    _, col, _ = st.sidebar.columns([0.3, 0.5, 0.2])

    if col.button("Analyze Flower") :
        with st.spinner(text="In progress..."):
            flower, acc = predict(image)

            st.markdown("<hr/>", unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.subheader("Answer ✅:")

            col1, col2 = st.columns([0.5, 0.5])
            col1.success(flower)
            col2.success(acc)

with header :
    _, col1, _ = st.columns([0.1,1,0.1])
    col1.title("Flower Classification App 🌺")

    st.markdown("<hr/>", unsafe_allow_html=True)

    
with body :
    st.subheader("Prediction of a Flower Class given a Flower Image, between 102 Different Classes")

    st.write("Upload the Flower Image in the Sidebar. The Model was trained using Transfer Learning, Google Colab GPU and RestNet50.")
