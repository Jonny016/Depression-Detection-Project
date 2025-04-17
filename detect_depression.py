import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import os

# Configuration
CONFIDENCE_THRESHOLD = 0.8 # Minimum confidence to make a prediction
MIN_FACE_SIZE = (60, 60)
MODEL_PATH = 'depression_model.h5'


# Load pre-trained model with error handling
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Error loading face cascade classifier!")
except Exception as e:
    print(f"Error initializing: {str(e)}")
    exit(1)

# Emotion labels
labels = ['Not Depressed', 'Depressed']

def validate_image(image):
    """Validate input image"""
    if image is None or image.size == 0:
        return False
    if len(image.shape) != 2:  # Check if grayscale
        return False
    return True

def preprocess_face(face):
    """Preprocess face with error handling"""
    try:
        if not validate_image(face):
            raise ValueError("Invalid face image")
        
        # Ensure minimum face size
        if face.shape[0] < 48 or face.shape[1] < 48:
            raise ValueError("Face too small for accurate detection")
            
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.astype('float') / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        return face
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def predict_depression(face):
    """Make prediction with error handling"""
    try:
        prediction = model.predict(face, verbose=0)[0]
        confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        
        # Always return a definitive prediction
        label = labels[predicted_class]
        
        # Adjust color intensity based on confidence
        if label == 'Not Depressed':
            color = (0, int(255 * confidence), 0)  # Green with varying intensity
        else:
            color = (0, 0, int(255 * confidence))  # Red with varying intensity
            
        return label, confidence, color
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, 0, None

# Initialize video capture
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video capture device!")
except Exception as e:
    print(f"Error opening camera: {str(e)}")
    exit(1)

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=MIN_FACE_SIZE)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            processed_face = preprocess_face(face)
            
            if processed_face is not None:
                label, confidence, color = predict_depression(processed_face)
                
                if label and color:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Display label with confidence
                    text = f"{label} ({confidence:.2f})"
                    cv2.putText(frame, text, (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        cv2.imshow('Depression Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        continue

cap.release()
cv2.destroyAllWindows()
