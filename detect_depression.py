import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
from collections import deque

# Configuration
MODEL_PATH = 'depression_model.h5'
MIN_FACE_SIZE = (30, 30)
CONFIDENCE_THRESHOLD = 0.65
PREDICTION_QUEUE_SIZE = 30  # Number of frames to consider for smoothing
STABILITY_THRESHOLD = 0.7   # Minimum ratio of consistent predictions to change state

# Load pre-trained models with error handling
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found!")
    model = load_model(MODEL_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    if face_cascade.empty() or smile_cascade.empty() or eye_cascade.empty():
        raise Exception("Error loading cascade classifiers!")
except Exception as e:
    print(f"Error initializing: {str(e)}")
    exit(1)

# Emotion labels
labels = ['Not Depressed', 'Depressed']

# Initialize prediction history
prediction_history = deque(maxlen=PREDICTION_QUEUE_SIZE)
current_stable_prediction = None
frames_since_change = 0

def get_stable_prediction(predictions):
    """Calculate stable prediction from history"""
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
    
    if max_count / total_frames >= STABILITY_THRESHOLD:
        # Get the prediction with highest count
        stable_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
        # Calculate average confidence for the stable prediction
        avg_conf = conf_sums[stable_pred] / pred_counts[stable_pred]
        return stable_pred, avg_conf
    
    return None, 0.0

def detect_facial_features(face_gray):
    """Detect multiple facial features"""
    try:
        # Detect smiles with adjusted parameters
        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.7,
            minNeighbors=15,
            minSize=(25, 25)
        )
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20)
        )
        
        features = {
            'has_smile': len(smiles) > 0,
            'smile_confidence': len(smiles) / 2 if len(smiles) <= 2 else 1.0,
            'has_eyes': len(eyes) >= 2,
            'eye_symmetry': abs(len(eyes) - 2) == 0
        }
        return features
    except:
        return {'has_smile': False, 'smile_confidence': 0.0, 'has_eyes': False, 'eye_symmetry': False}

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
        
        if face.shape[0] < 48 or face.shape[1] < 48:
            raise ValueError("Face too small for accurate detection")
            
        # Enhance image quality
        face = cv2.equalizeHist(face)  # Improve contrast
        face = cv2.resize(face, (48, 48))
        face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face = face.astype('float') / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)
        return face
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def predict_depression(face_array, face_gray):
    """Make prediction with improved accuracy and stability"""
    try:
        # Get model prediction
        prediction = model.predict(face_array, verbose=0)[0]
        base_confidence = np.max(prediction)
        predicted_class = np.argmax(prediction)
        
        # Get facial features
        features = detect_facial_features(face_gray)
        
        # Calculate feature-based confidence adjustment
        feature_confidence = 0.0
        if features['has_smile']:
            feature_confidence += features['smile_confidence'] * 0.3
        if features['has_eyes'] and features['eye_symmetry']:
            feature_confidence += 0.2
            
        # Blend model confidence with feature confidence
        confidence = (base_confidence * 0.7) + (feature_confidence * 0.3)
        
        # Determine initial prediction
        if confidence < CONFIDENCE_THRESHOLD:
            if features['has_smile'] and features['smile_confidence'] > 0.7:
                predicted_class = 0
            elif not features['has_smile'] and base_confidence > 0.6:
                predicted_class = 1
        
        label = labels[predicted_class]
        
        # Add to prediction history
        prediction_history.append((label, confidence))
        
        # Get stable prediction
        stable_label, stable_confidence = get_stable_prediction(prediction_history)
        
        if stable_label:
            # Use the stable prediction
            label = stable_label
            confidence = stable_confidence
        
        # Determine color based on confidence and prediction
        if label == 'Not Depressed':
            color = (0, int(255 * confidence), 0)
        else:
            color = (0, 0, int(255 * confidence))
            
        return label, confidence, color, features['has_smile']
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return None, 0, None, False

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
            face_gray = gray[y:y+h, x:x+w]
            processed_face = preprocess_face(face_gray)
            
            if processed_face is not None:
                label, confidence, color, has_smile = predict_depression(processed_face, face_gray)
                
                if label and color:
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Display label with confidence and smile indicator
                    smile_indicator = "😊" if has_smile else ""
                    text = f"{label} ({confidence:.2f}) {smile_indicator}"
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
