import json
import time
import numpy as np
import os
import cv2
import streamlit as st

# Import the team's AI models directly into the calibration tool
from src.vision.face_mesh import get_landmarks
from src.vision.eye_ear import compute_ear, LEFT_EYE, RIGHT_EYE
from src.vision.mouth_mar import compute_mar

# Ensure the profiles directory exists
os.makedirs('data/profiles', exist_ok=True)

def run_calibration(driver_name, duration_sec=5):
    """Records driver baselines using the live webcam and saves to JSON."""
    ear_list = []
    mar_list = []
    
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # The magic Windows un-freezer!
    start_time = time.time()
    
    # Create a placeholder in the sidebar to show the video feed
    video_placeholder = st.sidebar.empty()
    
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
    cap.release()
    video_placeholder.empty() # Clear the video when done
    
    # Safe fallback if the camera couldn't find a face
    if not ear_list:
        ear_list = [0.25]
        mar_list = [0.50]
        
    # Compute mean and standard deviation
    data = {
        'driver': driver_name,
        'ear_mean': float(np.mean(ear_list)),
        'ear_std': float(np.std(ear_list)),
        'mar_mean': float(np.mean(mar_list)),
        'yaw_mean': 0.0 # Placeholder for pose math later
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