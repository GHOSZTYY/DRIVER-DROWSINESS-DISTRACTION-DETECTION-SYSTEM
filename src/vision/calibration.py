import json
import time
import numpy as np
import os

# Ensure the profiles directory exists
os.makedirs('data/profiles', exist_ok=True)

def run_calibration(driver_name, get_ear_fn, get_mar_fn, get_pose_fn, duration_sec=60):
    """Records driver baselines for a set duration and saves to JSON."""
    ear_list = []
    mar_list = []
    yaw_list = []
    
    start_time = time.time()
    
    # Collect data for the specified duration
    while time.time() - start_time < duration_sec:
        # In a real Streamlit app, we might need to yield or handle frames differently,
        # but this follows the core logic of sampling the functions.
        ear_list.append(get_ear_fn())
        mar_list.append(get_mar_fn())
        yaw_list.append(get_pose_fn())
        
        time.sleep(0.1) # Small sleep to simulate frame timing and not overload CPU
        
    # Compute mean and standard deviation
    data = {
        'driver': driver_name,
        'ear_mean': float(np.mean(ear_list)),
        'ear_std': float(np.std(ear_list)),
        'mar_mean': float(np.mean(mar_list)),
        'yaw_mean': float(np.mean(yaw_list))
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