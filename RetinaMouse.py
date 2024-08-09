import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Blink detection variables
blink_threshold = 0.01
blink_frames = 0
blink_detected = False

def get_eye_aspect_ratio(landmarks, eye_indices):
    eye = [landmarks[i] for i in eye_indices]
    eye = np.array([(p.x, p.y) for p in eye])
    vertical_1 = np.linalg.norm(eye[1] - eye[5])
    vertical_2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame color to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get facial landmarks
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Get left and right eye landmarks
            left_eye_indices = [33, 7, 163, 144, 145, 153]
            right_eye_indices = [362, 382, 381, 380, 374, 373]

            # Calculate the aspect ratio for both eyes
            left_ear = get_eye_aspect_ratio(face_landmarks.landmark, left_eye_indices)
            right_ear = get_eye_aspect_ratio(face_landmarks.landmark, right_eye_indices)
            ear = (left_ear + right_ear) / 2.0

            # Calculate average eye coordinates for cursor movement
            left_eye_center = np.mean([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in left_eye_indices], axis=0)
            right_eye_center = np.mean([(face_landmarks.landmark[i].x * w, face_landmarks.landmark[i].y * h) for i in right_eye_indices], axis=0)
            eye_center = (left_eye_center + right_eye_center) / 2.0

            # Move the mouse cursor
            screen_x = np.clip(eye_center[0], 0, w) * screen_width / w
            screen_y = np.clip(eye_center[1], 0, h) * screen_height / h
            pyautogui.moveTo(screen_x, screen_y)

            # Blink detection
            if ear < blink_threshold:
                blink_frames += 1
            else:
                if blink_frames > 1:
                    blink_detected = True
                blink_frames = 0

            if blink_detected:
                pyautogui.click()
                blink_detected = False

    cv2.imshow('Eye Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()