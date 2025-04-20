import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import gdown
import os

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Constants
labels = ['Not Depressed', 'Depressed']
MIN_FACE_SIZE = (60, 60)

# Title
st.title("🧠 Depression Detection App")

# 📥 Download Model from Google Drive
@st.cache_resource
def download_model():
    model_url = "https://drive.google.com/uc?id=17oqp2bazaHwfa5BuI8lUleO19gV4MnCI"
    output_path = "model.h5"
    if not os.path.exists(output_path):
        gdown.download(model_url, output_path, quiet=False)
    return load_model(output_path)

# 📦 Load Haar Cascade
@st.cache_resource
def load_cascade():
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# 🔄 Face Preprocessing
def preprocess_face(face):
    try:
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        return face
    except:
        return None

# 🧠 Prediction
def predict_depression(model, face):
    prediction = model.predict(face, verbose=0)[0]
    confidence = np.max(prediction)
    label = labels[np.argmax(prediction)]
    return label, confidence

# 🎥 WebRTC Video Processing
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = load_cascade()
        self.model = download_model()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=MIN_FACE_SIZE)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            processed_face = preprocess_face(face)

            if processed_face is not None:
                label, confidence = predict_depression(self.model, processed_face)
                color = (0, 255, 0) if label == "Not Depressed" else (0, 0, 255)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                cv2.putText(img, f"{label} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img

# ⏯️ Webcam Stream
st.header("📷 Enable Webcam")
webrtc_streamer(key="depression-detection", video_processor_factory=VideoProcessor)
