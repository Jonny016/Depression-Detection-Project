import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import os
from collections import deque

# Configuration
CONFIDENCE_THRESHOLD = 0.6
MIN_FACE_SIZE = (60, 60)
MODEL_PATH = 'depression_model.h5'
SMOOTHING_FRAMES = 30  # Increased number of frames for smoothing
SMILE_THRESHOLD = 0.4  # Further increased threshold for smile detection confidence
MODEL_BIAS = 0.3  # Increased bias factor for depression
STABILITY_THRESHOLD = 0.6  # Lowered threshold for stable predictions
NEUTRAL_BIAS = 0.2  # Additional bias for neutral expressions

# Load pre-trained model with error handling
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
prediction_history = deque(maxlen=SMOOTHING_FRAMES)
current_stable_prediction = None

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
    
    if max_count / total_frames >= STABILITY_THRESHOLD:  # 80% agreement required
        stable_pred = max(pred_counts.items(), key=lambda x: x[1])[0]
        avg_conf = conf_sums[stable_pred] / pred_counts[stable_pred]
        return stable_pred, avg_conf
    
    return None, 0.0

def detect_facial_features(face_gray):
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
        # Get facial features first
        features = detect_facial_features(face_gray)
        
        # If smile is detected, immediately classify as Not Depressed
        if features['has_smile'] and features['smile_confidence'] > SMILE_THRESHOLD:
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
        prediction = model.predict(face_array, verbose=0)[0]
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
            depression_indicators += 2  # Increased weight for neutral expression
            
        # Apply bias to model prediction
        if predicted_class == 0:  # Not Depressed
            # Reduce confidence for "Not Depressed" predictions
            base_confidence = max(0.1, base_confidence - MODEL_BIAS)
        else:  # Depressed
            # Increase confidence for "Depressed" predictions
            base_confidence = min(1.0, base_confidence + MODEL_BIAS)
            
        # Use model prediction as primary decision
        label = labels[predicted_class]
        confidence = base_confidence
        
        # Override prediction for neutral or sad faces
        if features['is_neutral'] or depression_indicators >= 2:
            label = 'Depressed'
            confidence = max(confidence, 0.7 + NEUTRAL_BIAS)  # Increased confidence for neutral faces
        
        # Adjust confidence based on facial features
        if label == 'Depressed' and depression_indicators >= 2:
            confidence = min(confidence + 0.2, 1.0)  # Increased boost for multiple indicators
        
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
