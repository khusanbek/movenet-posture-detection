
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Load MoveNet (Lightning or Thunder)
movenet = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

def detect_keypoints(image):
    image = tf.image.resize_with_pad(tf.convert_to_tensor(image), 192, 192)
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.cast(input_image, dtype=tf.int32)

    outputs = movenet.signatures["serving_default"](input_image)
    keypoints = outputs["output_0"].numpy()[0, 0, :, :]

    return keypoints  # List of 17 keypoints

def calculate_angle(a, b, c):
    """Calculate angle between three keypoints."""
    a, b, c = np.array(a[:2]), np.array(b[:2]), np.array(c[:2])
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(cosine_angle))

def detect_posture(keypoints):
    shoulder_mid = (keypoints[5][:2] + keypoints[6][:2]) / 2
    hip_mid = (keypoints[11][:2] + keypoints[12][:2]) / 2
    knee_mid = (keypoints[13][:2] + keypoints[14][:2]) / 2

    back_angle = calculate_angle(keypoints[5], hip_mid, keypoints[11])
    neck_angle = calculate_angle(keypoints[0], shoulder_mid, keypoints[5])

    if back_angle > 160 and neck_angle > 160:
        return "Standing Straight"
    elif back_angle < 140:
        return "Slouching"
    elif hip_mid[1] > knee_mid[1]:
        return "Sitting"
    else:
        return "Unknown Posture"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = detect_keypoints(frame)
    posture = detect_posture(keypoints)

    # Display posture on frame
    cv2.putText(frame, posture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('MoveNet Posture Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
