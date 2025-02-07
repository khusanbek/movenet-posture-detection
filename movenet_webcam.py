
import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load MoveNet Thunder model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

# Define function to process frames
def detect_pose(frame):
    h, w, _ = frame.shape
    input_image = cv2.resize(frame, (256, 256))
    input_image = np.expand_dims(input_image, axis=0).astype(np.int32)

    # Run MoveNet inference
    outputs = model.signatures['serving_default'](tf.constant(input_image))
    keypoints = outputs['output_0'].numpy()[0][0]

    # Draw keypoints on frame
    for kp in keypoints:
        y, x, confidence = kp
        if confidence > 0.3:  # Only draw points with high confidence
            x, y = int(x * w), int(y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    return frame

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    frame = detect_pose(frame)

    cv2.imshow("MoveNet Pose Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()