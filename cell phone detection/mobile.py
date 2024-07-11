import cv2
from ultralytics import YOLO
import winsound

# Load the pre-trained YOLOv8 model
model = YOLO(r'F:\cell phone\yolov8n.pt')  # Pre-trained YOLOv8n model

# Class ID for mobile phones
MOBILE_PHONE_CLASS_ID = 67

# Define the frequency and duration for the alert sound
ALERT_SOUND_FREQUENCY = 1000  # Frequency in Hz
ALERT_SOUND_DURATION = 500  # Duration in milliseconds


# Define a function to play sound
def play_sound():
    winsound.Beep(ALERT_SOUND_FREQUENCY, ALERT_SOUND_DURATION)


# Define a function to process each frame
def process_frame(frame):
    results = model(frame)  # Perform detection

    detected_phone = False  # Flag to track if a phone has been detected

    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            if class_id == MOBILE_PHONE_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_name = model.names[class_id]

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name} {confidence:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Set flag to True to indicate that a phone is detected
                detected_phone = True

    # Play sound if a phone is detected
    if detected_phone:
        play_sound()

    return frame


# Initialize the video capture object
cap = cv2.VideoCapture(0)  # Use the correct camera index if you have multiple cameras

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Object Detection', processed_frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
