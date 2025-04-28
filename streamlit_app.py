import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import gdown
import os
from collections import deque

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# Constants
labels = ['Not Depressed', 'Depressed']
MIN_FACE_SIZE = (60, 60)
SMOOTHING_FRAMES = 30  # Increased number of frames for smoothing
SMILE_THRESHOLD = 0.3  # Increased threshold for smile detection confidence
MODEL_BIAS = 0.2  # Increased bias factor
STABILITY_THRESHOLD = 0.7  # Lowered threshold for stable predictions

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
    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_smile.xml')
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, smile_cascade, eye_cascade

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

# 👁️ Detect Facial Features
def detect_facial_features(face_gray, smile_cascade, eye_cascade):
    """Detect multiple facial features"""
    try:
        # Detect smiles with extremely sensitive parameters
        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,  # Extremely sensitive scaling
            minNeighbors=3,  # Very low for better detection
            minSize=(8, 8)  # Very small minimum size
        )
        
        # Detect eyes with more sensitive parameters
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=4,
            minSize=(15, 15)
        )
        
        # Calculate facial features
        has_smile = len(smiles) > 0
        smile_confidence = min(len(smiles) / 2, 1.0) if len(smiles) <= 2 else 1.0
        
        # Check for droopy eyes (potential sign of depression)
        droopy_eyes = False
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda x: x[0])
            eye_distance = abs(eyes[1][0] - eyes[0][0])
            face_width = face_gray.shape[1]
            if eye_distance < face_width * 0.25:  # More sensitive to close eyes
                droopy_eyes = True
        
        # Check for neutral expression (no strong emotion)
        is_neutral = not has_smile and not droopy_eyes
        
        return {
            'has_smile': has_smile,
            'smile_confidence': smile_confidence,
            'has_eyes': len(eyes) >= 2,
            'droopy_eyes': droopy_eyes,
            'is_neutral': is_neutral
        }
    except:
        return {
            'has_smile': False,
            'smile_confidence': 0.0,
            'has_eyes': False,
            'droopy_eyes': False,
            'is_neutral': True
        }

# 🔄 Get Stable Prediction
def get_stable_prediction(predictions):
    """Calculate stable prediction from history with increased stability"""
    if not predictions:
        return None, 0.0
    
    # Count occurrences of each prediction
    pred_counts = {'Not Depressed': 0, 'Depressed': 0}
    conf_sums = {'Not Depressed': 0.0, 'Depressed': 0.0}
    
    for pred, conf in predictions:
        pred_counts[pred] += 1
        conf_sums[pred] += conf
    
    # Find the most common prediction
    max_count = max(pred_counts.values())
    total_frames = len(predictions)
    
    if max_count / total_frames >= STABILITY_THRESHOLD:  # 70% agreement required
        stable_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
        avg_conf = conf_sums[stable_pred] / pred_counts[stable_pred]
        return stable_pred, avg_conf
    
    return None, 0.0

# 🧠 Prediction
def predict_depression(model, face, face_gray, smile_cascade, eye_cascade, prediction_history):
    """Make prediction with improved accuracy and stability"""
    try:
        # Get facial features first
        features = detect_facial_features(face_gray, smile_cascade, eye_cascade)
        
        # If smile is detected, immediately classify as Not Depressed
        if features['has_smile']:
            label = 'Not Depressed'
            confidence = 0.8  # High confidence for smiles
            color = (0, int(255 * confidence), 0)  # Green with varying intensity
            
            # Add to prediction history
            prediction_history.append((label, confidence))
            
            # Get stable prediction
            stable_label, stable_confidence = get_stable_prediction(prediction_history)
            
            if stable_label:
                # Use the stable prediction
                label = stable_label
                confidence = stable_confidence
                color = (0, int(255 * confidence), 0) if label == 'Not Depressed' else (0, 0, int(255 * confidence))
            
            return label, confidence, color, features['has_smile']
        
        # If no smile, proceed with model prediction
        prediction = model.predict(face, verbose=0)[0]
        base_confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        
        # Calculate depression indicators
        depression_indicators = 0
        if not features['has_smile']:
            depression_indicators += 1
        if features['droopy_eyes']:
            depression_indicators += 1
        if not features['has_eyes']:
            depression_indicators += 1
        if features['is_neutral']:
            depression_indicators += 1
            
        # Apply slight bias to model prediction
        if predicted_class == 0:  # Not Depressed
            # Slightly reduce confidence for "Not Depressed" predictions
            base_confidence = max(0.1, base_confidence - MODEL_BIAS)
        else:  # Depressed
            # Slightly increase confidence for "Depressed" predictions
            base_confidence = min(1.0, base_confidence + MODEL_BIAS)
            
        # Use model prediction as primary decision
        label = labels[predicted_class]
        confidence = base_confidence
        
        # Override prediction for neutral or sad faces
        if features['is_neutral'] or depression_indicators >= 2:
            label = 'Depressed'
            confidence = max(confidence, 0.7)
        
        # Adjust confidence based on facial features
        if label == 'Depressed' and depression_indicators >= 2:
            confidence = min(confidence + 0.1, 1.0)  # Boost confidence if multiple indicators
        
        # Add to prediction history
        prediction_history.append((label, confidence))
        
        # Get stable prediction
        stable_label, stable_confidence = get_stable_prediction(prediction_history)
        
        if stable_label:
            # Use the stable prediction
            label = stable_label
            confidence = stable_confidence
        
        # Determine color based on prediction
        if label == 'Not Depressed':
            color = (0, int(255 * confidence), 0)  # Green with varying intensity
        else:
            color = (0, 0, int(255 * confidence))  # Red with varying intensity
            
        return label, confidence, color, features['has_smile']
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, 0, None, False

# 🎥 WebRTC Video Processing
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.face_cascade, self.smile_cascade, self.eye_cascade = load_cascade()
        self.model = download_model()
        self.prediction_history = deque(maxlen=SMOOTHING_FRAMES)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=MIN_FACE_SIZE)

        for (x, y, w, h) in faces:
            face_gray = gray[y:y+h, x:x+w]
            processed_face = preprocess_face(face_gray)

            if processed_face is not None:
                label, confidence, color, has_smile = predict_depression(
                    self.model, processed_face, face_gray, 
                    self.smile_cascade, self.eye_cascade, self.prediction_history
                )
                
                if label and color:
                    # Draw rectangle around face
                    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                    
                    # Display label with confidence and smile indicator
                    smile_indicator = "😊" if has_smile else ""
                    text = f"{label} ({confidence:.2f}) {smile_indicator}"
                    cv2.putText(img, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return img

# ⏯️ Webcam Stream
st.header("📷 Enable Webcam")
webrtc_streamer(key="depression-detection", video_processor_factory=VideoProcessor)
