import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# --- THE MONKEY PATCH ---
# DeepFace throws a tantrum if it can't find the version string.
# We are manually injecting it here before DeepFace even gets imported!
import tensorflow as tf
if not hasattr(tf, "__version__"):
    tf.__version__ = "2.16.1"
# ------------------------

import cv2
import time
import threading

from src.alerts.sos import send_sos_alert
from src.alerts.beep import play_beep
from src.alerts.voice import speak_warning
from src.vision.calibration import run_calibration, load_profile
from src.data.db import create_tables, start_session, log_event, end_session

# --- IMPORT TEAM AI MODELS ---
from src.vision.face_mesh import get_landmarks
from src.vision.eye_ear import compute_ear, LEFT_EYE, RIGHT_EYE
from src.vision.mouth_mar import compute_mar
from src.vision.head_pose import get_pose
from src.brain.emotion_detection import detect_emotion
from src.brain.phone_detector import PhoneDetector
from src.brain.fusion import calculate_driver_state

WINDOW_NAME = "Driver Safety Co-Pilot"
DROWSY_THRESHOLD_FRAMES = 15  # ~0.5s at 30fps of eyes-below-baseline before flagging drowsy
EMOTION_EVERY_N_FRAMES = 15   # DeepFace is heavy — throttle it

# level names used by fusion.py's status -> beep.py's sound keys
STATUS_TO_BEEP_LEVEL = {
    "WARNING": "low",
    "DROWSY": "medium",
    "CRITICAL": "critical",
}

_voice_thread = None


def speak_async(message):
    """Fire voice alert on a background thread so it never blocks the camera loop."""
    global _voice_thread
    if _voice_thread is not None and _voice_thread.is_alive():
        return  # already speaking, don't stack requests
    _voice_thread = threading.Thread(target=speak_warning, args=(message,), daemon=True)
    _voice_thread.start()


def sos_async(driver_name):
    """Twilio call is a blocking network request — fire it off-thread."""
    threading.Thread(target=send_sos_alert, args=(driver_name,), daemon=True).start()


def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def draw_hud(frame, profile, state, current_emotion, total_yawns, fps):
    y = 30
    cv2.putText(frame, f"Driver: {profile['driver']}  FPS: {fps}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y += 25
    color = (0, 255, 0) if state["status"] == "SAFE" else (0, 165, 255) if state["status"] == "WARNING" else (0, 0, 255)
    cv2.putText(frame, f"Status: {state['status']}  Score: {state['score']}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    y += 25
    if state["reasons"]:
        cv2.putText(frame, "Reasons: " + ", ".join(state["reasons"]), (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        y += 25
    cv2.putText(frame, f"Emotion: {current_emotion}  Total yawns: {total_yawns}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
    return frame


def run_driving_loop(driver_name, profile, cap):
    session_id = start_session(driver_name)
    total_alerts = 0
    total_yawns = 0
    is_yawning = False
    drowsy_frames = 0
    current_emotion = "Neutral"
    prev_status = "SAFE"
    frame_count = 0
    prev_time = time.time()
    fps = 0

    phone_tracker = PhoneDetector(confidence_threshold=0.5, frame_threshold=10)

    print("Co-Pilot active. Press 'q' to stop driving.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera disconnected.")
            break

        frame_count += 1

        # ==========================================
        # 1. RUN THE AI MODELS
        # ==========================================
        landmarks = get_landmarks(frame)

        if landmarks:
            left = compute_ear(landmarks, LEFT_EYE)
            right = compute_ear(landmarks, RIGHT_EYE)
            current_ear = (left + right) / 2.0
            current_mar = compute_mar(landmarks)
            pose_result = get_pose(landmarks, frame.shape)
        else:
            current_ear = profile['ear_mean']
            current_mar = profile['mar_mean']
            pose_result = None

        if frame_count % EMOTION_EVERY_N_FRAMES == 0:
            emotion_data = detect_emotion(frame)
            if emotion_data["emotion"]:
                current_emotion = emotion_data["emotion"]

        # Phone detection: no real object-detection model wired in yet —
        # kept as an always-empty placeholder on purpose (tracked separately).
        raw_detections = []
        phone_result = phone_tracker.detect(raw_detections)

        # ==========================================
        # 2. PERSONALIZED EYE / YAWN CHECKS (against calibrated baseline)
        # ==========================================
        if current_ear < (profile['ear_mean'] * 0.8):
            drowsy_frames += 1
        else:
            drowsy_frames = 0
        eye_result = {"ear": current_ear, "drowsy": drowsy_frames >= DROWSY_THRESHOLD_FRAMES}

        if current_mar > (profile['mar_mean'] * 1.5):
            if not is_yawning:
                total_yawns += 1
            is_yawning = True
        else:
            is_yawning = False
        yawn_result = {"mar": current_mar, "yawning": is_yawning}

        emotion_result = {"emotion": current_emotion}

        # ==========================================
        # 3. FUSION — single source of truth for the alert decision
        # ==========================================
        state = calculate_driver_state(eye_result, yawn_result, phone_result, pose_result, emotion_result)

        # Fire alerts only on a state change so sounds/voice/db don't spam every frame
        if state["status"] != prev_status:
            level = STATUS_TO_BEEP_LEVEL.get(state["status"])
            if level:
                play_beep(level)
            if state["alarm"]:
                total_alerts += 1
                log_event(session_id, level or "low", ear=current_ear, mar=current_mar,
                          yaw=pose_result["yaw"] if pose_result else None,
                          phone=phone_result["phone_detected"], emotion=current_emotion)
            if state["status"] == "CRITICAL":
                speak_async("Warning. Critical driver impairment detected. Please pull over.")
                sos_async(driver_name)
        prev_status = state["status"]

        # ==========================================
        # 4. VIDEO + HUD
        # ==========================================
        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time + 0.001))
        prev_time = curr_time

        draw_hud(frame, profile, state, current_emotion, total_yawns, fps)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    end_session(session_id, total_alerts)


def main():
    create_tables()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # resizable/maximizable — image auto-scales to window size

    driver_name = input("Driver name [Driver]: ").strip() or "Driver"
    cap = open_camera()
    if not cap.isOpened():
        print("Error: Could not open webcam (index 0). Is it in use by another app?")
        return

    profile = load_profile(driver_name)
    if profile and not profile.get("calibrated", True):
        print("⚠️  Existing profile was saved without a detected face — recalibrate for accurate results.")

    print(f"\n{'✅ Profile active' if profile else '⚠️  No profile found'} for '{driver_name}'.")
    print("Controls:  c = calibrate (5s)   s = start driving   q = quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera disconnected.")
                break

            label = f"Driver: {driver_name}  |  {'Profile OK' if profile else 'NOT CALIBRATED'}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "c = calibrate   s = start driving   q = quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                profile = run_calibration(driver_name, cap, duration_sec=5, window_name=WINDOW_NAME)
                print("Calibration complete." if profile.get("calibrated") else
                      "Calibration finished but no face was detected — using generic fallback values.")
            elif key == ord('s'):
                if not profile:
                    print("Crucial: you must calibrate your baseline before driving!")
                    continue
                run_driving_loop(driver_name, profile, cap)
                print("Stopped driving. Press 's' to resume, 'c' to recalibrate, or 'q' to quit.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
