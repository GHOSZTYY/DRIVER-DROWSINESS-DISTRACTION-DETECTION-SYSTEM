import json
import time
import numpy as np
import os
import cv2

# Import the team's AI models directly into the calibration tool
from src.vision.face_mesh import get_landmarks
from src.vision.eye_ear import compute_ear, LEFT_EYE, RIGHT_EYE
from src.vision.mouth_mar import compute_mar

# Ensure the profiles directory exists
os.makedirs('data/profiles', exist_ok=True)

def run_calibration(driver_name, cap, duration_sec=5, window_name="Driver Safety Co-Pilot"):
    """
    Records driver baselines using the given (already-open) webcam capture
    and saves them to JSON. Shows a live countdown in the given cv2 window
    so the driver knows calibration is in progress.
    """
    ear_list = []
    mar_list = []

    start_time = time.time()

    while time.time() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = get_landmarks(frame)
        if landmarks:
            left = compute_ear(landmarks, LEFT_EYE)
            right = compute_ear(landmarks, RIGHT_EYE)
            ear_list.append((left + right) / 2.0)
            mar_list.append(compute_mar(landmarks))

        # Show the video during calibration so you know it's working!
        remaining = duration_sec - (time.time() - start_time)
        cv2.putText(frame, f"Calibrating... {remaining:.1f}s  Look straight ahead, act naturally",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    # Safe fallback if the camera couldn't find a face
    calibrated = bool(ear_list)
    if not ear_list:
        ear_list = [0.25]
        mar_list = [0.50]

    # Compute mean and standard deviation
    data = {
        'driver': driver_name,
        'ear_mean': float(np.mean(ear_list)),
        'ear_std': float(np.std(ear_list)),
        'mar_mean': float(np.mean(mar_list)),
        'yaw_mean': 0.0, # Placeholder for pose math later
        'calibrated': calibrated # False if no face was ever detected during calibration
    }
    
    # Save as JSON
    file_path = f'data/profiles/{driver_name}.json'
    with open(file_path, 'w') as f:
        json.dump(data, f)
        
    return data

def load_profile(driver_name):
    """Reads the JSON profile and returns the dictionary."""
    file_path = f'data/profiles/{driver_name}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return None