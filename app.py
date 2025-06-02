import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- CONFIG ---
IMAGE_SIZE = 224  # Set to the size you used for training
MODEL_PATH = 'cnn_brain_tumor_model.h5'  # Path to your trained model
CLASS_LABELS = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cnn_brain_tumor_model.h5')

model = load_model()

# --- PREPROCESSING ---
def preprocess_img(img, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- STREAMLIT UI ---
st.title("🧠 Brain Tumor Classifier")
st.write("Upload an MRI image to predict tumor type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    with st.spinner("Predicting..."):
        preprocessed_img = preprocess_img(img)
        prediction = model.predict(preprocessed_img)[0]
        predicted_class = CLASS_LABELS[np.argmax(prediction)]
        confidence = np.max(prediction)

    st.success(f"🎯 Predicted Tumor Type: **{predicted_class}**")
    st.info(f"🧪 Confidence: {confidence*100:.2f}%")
