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

# ✅ Download model from Google Drive (with confirmation token)
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None

    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    with open(destination, 'wb') as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# ✅ Load model with caching and file validation
@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        download_file_from_google_drive(DRIVE_FILE_ID, MODEL_PATH)

        # Check if file is valid (avoid HTML download issue)
        if os.path.getsize(MODEL_PATH) < 5000000:  # Less than 5MB? Likely corrupted
            st.error("❌ Model download failed. File too small or invalid.")
            st.stop()

        st.success("✅ Model downloaded successfully!")

    return load_model(MODEL_PATH)

# Load model and classifier
model = download_and_load_model()
labels = ['Not Depressed', 'Depressed']
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# UI
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
        label = labels[predicted_class]
        color = (0, 255, 0) if label == 'Not Depressed' else (0, 0, 255)
        return label, confidence, color
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0, None

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=MIN_FACE_SIZE)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            processed_face = preprocess_face(face)
            if processed_face is not None:
                label, confidence, color = predict_depression(processed_face)
                if label and color:
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    text = f"{label} ({confidence:.2f})"
                    cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img

# Start webcam stream
webrtc_streamer(key="depression-stream", video_transformer_factory=VideoTransformer)
