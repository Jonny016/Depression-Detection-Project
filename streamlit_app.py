import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import os
import requests

# Configuration
CONFIDENCE_THRESHOLD = 0.8
MIN_FACE_SIZE = (60, 60)
MODEL_PATH = 'depression_model.h5'
DRIVE_FILE_ID = "1OfP9oDdP4mZKUa6BFCuN2bz_spxm2LpH"

# --- Fixed Drive Download Function with confirmation bypass ---
def download_file_from_google_drive(file_id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value
        return None

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# --- Load model with download fallback ---
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        download_file_from_google_drive(DRIVE_FILE_ID, MODEL_PATH)
        st.success("Model downloaded successfully!")
    return load_model(MODEL_PATH)

# Load model and cascade
model = download_and_load_model()
labels = ['Not Depressed', 'Depressed']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit UI
st.title("🧠 Real-Time Depression Detection")
st.markdown("Webcam-based depression detection using deep learning.")

def validate_image(image):
    return image is not None and image.size != 0 and len(image.shape) == 2

def preprocess_face(face):
    try:
        if not validate_image(face):
            return None
        if face.shape[0] < 48 or face.shape[1] < 48:
            return None
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.astype("float32") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        return face
    except:
        return None

def predict_depression(face):
    try:
        prediction = model.predict(face, verbose=0)[0]
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        if confidence < CONFIDENCE_THRESHOLD:
            return "Uncertain", confidence, (128, 128, 128)
        label
