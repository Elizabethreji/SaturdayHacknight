import cv2
import numpy as np

def find_dominant_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    max_coords = np.unravel_index(hist.argmax(), hist.shape)
    dominant_color_hue = max_coords[0]
    dominant_color_saturation = max_coords[1]
    return dominant_color_hue, dominant_color_saturation

def classify_gesture(hue, saturation):
    if 10 <= hue <= 30 and 30 <= saturation <= 150:
        return "Rock"
    elif 60 <= hue <= 120 and 60 <= saturation <= 180:
        return "Paper"
    elif 0 <= hue <= 12 and 80 <= saturation <= 220:
        return "Scissors"
    else:
        return "Unknown"

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Ensure that the frame is valid
    if not ret:
        print("Error: Unable to capture frame.")
        break

    # Display the countdown
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(3, 0, -1):
        cv2.putText(frame, f"Game starting in {i}", (100, 50),
                    font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Rock Paper Scissors", frame)
        cv2.waitKey(1000)

    # Extract the ROI
    roi = frame[100:300, 100:300]

    # Check if the ROI is valid
    if roi.size == 0:
        print("Error: ROI is empty.")
        break

    # Find the dominant color in the ROI
    hue, saturation = find_dominant_color(roi)

    # Classify the gesture based on the dominant color
    gesture = classify_gesture(hue, saturation)

    # Display the recognized gesture
    cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Rock Paper Scissors", frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
