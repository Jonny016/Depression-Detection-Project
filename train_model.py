import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import cv2
import glob

# Configuration
IMG_SIZE = 48
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001

def create_model():
    model = Sequential([
        # First Convolutional Block - Feature Detection
        Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Second Convolutional Block - Feature Extraction
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Third Convolutional Block - Deep Features
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        # Fourth Convolutional Block - Fine Details
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        # Dense Layers - Classification
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')  # 2 classes: Not Depressed, Depressed
    ])

    return model

def load_and_preprocess_data(data_dir):
    images = []
    labels = []
    
    # Load Not Depressed images
    not_depressed_path = os.path.join(data_dir, 'not_depressed', '*')
    for img_path in glob.glob(not_depressed_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(0)  # 0 for Not Depressed

    # Load Depressed images
    depressed_path = os.path.join(data_dir, 'depressed', '*')
    for img_path in glob.glob(depressed_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(1)  # 1 for Depressed

    return np.array(images), np.array(labels)

def main():
    # Create data directories if they don't exist
    os.makedirs('data/not_depressed', exist_ok=True)
    os.makedirs('data/depressed', exist_ok=True)

    print("Please ensure you have placed your training images in the following directories:")
    print("- data/not_depressed/: for images of people who are not depressed")
    print("- data/depressed/: for images of people who are depressed")
    
    input("Press Enter to continue once you've organized your data...")

    # Load and preprocess data
    print("Loading and preprocessing data...")
    images, labels = load_and_preprocess_data('data')

    if len(images) == 0:
        print("No images found! Please add images to the data directories.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

    # Calculate class weights to handle imbalanced data
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        rescale=1./255
    )

    # Create and compile model
    print("Creating and compiling model...")
    model = create_model()
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train model
    print("Training model...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test/255.0, y_test),
        class_weight=class_weight_dict,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5
            )
        ]
    )

    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test/255.0, y_test)
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    # Save model
    print("Saving model...")
    model.save('depression_model.h5')
    print("Model saved as 'depression_model.h5'")

if __name__ == "__main__":
    main()
