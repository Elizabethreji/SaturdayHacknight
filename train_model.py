import tensorflow as tf
from tensorflow.keras import layers, models

# Load and preprocess the dataset (replace with your actual dataset loading code)
# X_train, y_train = load_and_preprocess_data()

# Define the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model (replace with your actual training code)
# model.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Save the trained model
model.save('rps_model.h5')
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('rps_model.h5')

def classify_gesture(frame):
    # ... existing code to preprocess the frame

    prediction = model.predict(reshaped_frame)
    predicted_class = np.argmax(prediction)

    gestures = ["Rock", "Paper", "Scissors"]
    return gestures[predicted_class]

# ... rest of the OpenCV script
