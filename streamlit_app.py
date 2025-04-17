import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import tempfile

# Constants
labels = ['Not Depressed', 'Depressed']
MIN_FACE_SIZE = (60, 60)

# File uploader for the model
st.title("🧠 Depression Detection App")
uploaded_model = st.file_uploader("Upload your trained depression model (.h5)", type=["h5"])

# Load Haar Cascade
@st.cache_resource
def load_cascade():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

face_cascade = load_cascade()

# Function to preprocess face
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

# Function to predict
def predict_depression(model, face):
    prediction = model.predict(face, verbose=0)[0]
    confidence = np.max(prediction)
    label = labels[np.argmax(prediction)]
    return label, confidence

# App UI
if uploaded_model:
    # Save the uploaded model to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
        tmp_file.write(uploaded_model.read())
        tmp_model_path = tmp_file.name
        model = load_model(tmp_model_path)

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Webcam not working")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=MIN_FACE_SIZE)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                processed_face = preprocess_face(face)

                if processed_face is not None:
                    label, confidence = predict_depression(model, processed_face)
                    color = (0, 255, 0) if label == "Not Depressed" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
else:
    st.info("👆 Upload your `.h5` model to get started.")
