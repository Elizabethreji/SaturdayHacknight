import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('rps_model.h5')

# Function to classify the gesture
def classify_gesture(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 3))

    prediction = model.predict(reshaped_frame)
    predicted_class = np.argmax(prediction)

    gestures = ["Rock", "Paper", "Scissors"]
    return gestures[predicted_class]

# Open a connection to the webcam (0 represents the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow("Rock Paper Scissors", frame)

    # Classify the gesture
    player_gesture = classify_gesture(frame)

    # Display the recognized gesture
    print(f"Your Gesture: {player_gesture}")

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
# Inside the classify_gesture function, add print statements
def classify_gesture(frame):
    resized_frame = cv2.resize(frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 64, 64, 3))

    # Add print statements to debug
    print("Reshaped Frame Shape:", reshaped_frame.shape)
    print("Normalized Frame Values:", normalized_frame)

    prediction = model.predict(reshaped_frame)
    predicted_class = np.argmax(prediction)

    gestures = ["Rock", "Paper", "Scissors"]
    print("Predicted Class:", predicted_class, "Gesture:", gestures[predicted_class])
    return gestures[predicted_class]
