import json
import unicodedata
import time
import numpy as np
import os
import cv2

# Import the team's AI models directly into the calibration tool
from src.vision.face_mesh import get_landmarks
from src.vision.eye_ear import compute_ear, LEFT_EYE, RIGHT_EYE
from src.vision.mouth_mar import compute_mar
from src.vision.head_pose import get_pose
from src.vision.lighting import detect_face, eyes_covered

# Ensure the profiles directory exists
os.makedirs('data/profiles', exist_ok=True)

# Bumped whenever the measurements change meaning; older profiles must be redone.
PROFILE_VERSION = 4  # 4: mouth openness (MAR) now measured on the inner lips

# Calibration is only accepted if the face was seen in enough of the frames...
MIN_FACE_FRACTION = 0.6
MIN_FACE_FRAMES = 10
# ...and the eyes were really open. Open eyes measure about 0.22-0.30 EAR with
# these landmarks; closed eyes about 0.05-0.15. Below this the driver was
# probably squinting, half asleep or the eyes weren't visible.
MIN_OPEN_EAR = 0.15
# The head must stay fairly still while the straight-ahead angle is recorded.
MAX_POSE_STD_DEG = 8.0

RAW_POSE = {"pitch": 0.0, "yaw": 0.0}  # baseline that leaves the angles untouched

# The driver name becomes a file name (data/profiles/<name>.json), so only
# letters, digits, spaces, '-' and '_' are allowed. Anything else, such as
# '../' or '\\', could write or read files outside the profiles folder.
MAX_NAME_LENGTH = 40
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {f"{p}{n}" for p in ("COM", "LPT") for n in range(1, 10)}


def valid_driver_name(name):
    """True if the name is safe to use as a profile file name."""
    return (
        isinstance(name, str)
        and 0 < len(name) <= MAX_NAME_LENGTH
        and name == name.strip()
        # letters/digits in any script; category M = vowel signs and accents (e.g. Hindi)
        and all(ch.isalnum() or unicodedata.category(ch).startswith("M") or ch in " -_"
                for ch in name)
        and name.upper() not in _WINDOWS_RESERVED
    )


def _check_name(name):
    if not valid_driver_name(name):
        raise ValueError(f"invalid driver name {name!r}: use up to {MAX_NAME_LENGTH} "
                         "letters, digits, spaces, '-' or '_'")

def run_calibration(driver_name, cap, duration_sec=5, window_name="Driver Safety Co-Pilot"):
    """
    Records driver baselines using the given (already-open) webcam capture
    and saves them to JSON. Shows a live countdown in the given cv2 window
    so the driver knows calibration is in progress.

    Returns the profile dict. If the recording wasn't good enough,
    "calibrated" is False, "error" says why, and nothing is saved.
    """
    _check_name(driver_name)
    ear_list = []
    mar_list = []
    pitch_list = []
    yaw_list = []
    total_frames = 0
    covered_frames = 0

    start_time = time.time()

    while time.time() - start_time < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        landmarks, _ = detect_face(frame, get_landmarks)
        if landmarks and eyes_covered(frame, landmarks):
            covered_frames += 1
        if landmarks:
            left = compute_ear(landmarks, LEFT_EYE)
            right = compute_ear(landmarks, RIGHT_EYE)
            ear_list.append((left + right) / 2.0)
            mar_list.append(compute_mar(landmarks))
            pose = get_pose(landmarks, frame.shape, baseline=RAW_POSE)
            if pose:
                pitch_list.append(pose["pitch"])
                yaw_list.append(pose["yaw"])

        # Show the video during calibration so you know it's working!
        remaining = duration_sec - (time.time() - start_time)
        cv2.putText(frame, f"Calibrating... {remaining:.1f}s  Look straight ahead, act naturally",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Face found in {len(ear_list)}/{total_frames} frames",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(window_name, frame)
        cv2.waitKey(1)

    # Reject a bad recording instead of saving made-up fallback values
    error = None
    face_fraction = len(ear_list) / total_frames if total_frames else 0.0
    if len(ear_list) < MIN_FACE_FRAMES or face_fraction < MIN_FACE_FRACTION:
        error = (f"face found in only {len(ear_list)}/{total_frames} frames - "
                 "sit in front of the camera with your face fully visible")
    elif covered_frames > len(ear_list) / 2:
        error = "eyes are covered (sunglasses?) - remove them and try again"
    elif float(np.median(ear_list)) < MIN_OPEN_EAR:
        error = ("eyes looked closed or covered (sunglasses?) - "
                 "keep your eyes open normally and try again")
    elif len(pitch_list) < MIN_FACE_FRAMES:
        error = "head angle could not be measured - face the camera and try again"
    elif max(np.std(pitch_list), np.std(yaw_list)) > MAX_POSE_STD_DEG:
        error = "head moved too much - keep looking at the road and try again"
    if error:
        return {'driver': driver_name, 'calibrated': False, 'error': error}

    # Median instead of mean so blinks and talking during the 5 s don't drag
    # the baseline down/up. The *_mean keys are kept for the rest of the app.
    data = {
        'driver': driver_name,
        'version': PROFILE_VERSION,
        'ear_mean': float(np.median(ear_list)),
        'ear_std': float(np.std(ear_list)),
        'mar_mean': float(np.median(mar_list)),
        # the driver's own straight-ahead head angle for this camera position
        'pitch_mean': float(np.median(pitch_list)),
        'yaw_mean': float(np.median(yaw_list)),
        'face_frames': len(ear_list),
        'total_frames': total_frames,
        'calibrated': True
    }

    # Save as JSON
    file_path = f'data/profiles/{driver_name}.json'
    with open(file_path, 'w') as f:
        json.dump(data, f)
        
    return data

def load_profile(driver_name):
    """Reads the JSON profile and returns the dictionary.
    Returns None if there is no profile, or if it was made by an older
    version of the app or without a detected face (it must be redone)."""
    _check_name(driver_name)
    file_path = f'data/profiles/{driver_name}.json'
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        if data.get('version') != PROFILE_VERSION or not data.get('calibrated'):
            print(f"Profile for {driver_name} is outdated or incomplete - please recalibrate.")
            return None
        return data
    return None