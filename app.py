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
from src.vision.phone_yolo import get_detections
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
from src.brain.fusion import calculate_driver_state, PHONE_WARNING_SEC

WINDOW_NAME = "Driver Safety Co-Pilot"
EMOTION_EVERY_SEC = 2.0   # DeepFace is heavy — run it in the background this often

# level names used by fusion.py's status -> beep.py's sound keys
STATUS_TO_BEEP_LEVEL = {
    "WARNING": "low",
    "DANGER": "medium",
    "CRITICAL": "critical",
}

# Higher number = more serious; used to spot when things get worse
STATUS_RANK = {"SAFE": 0, "WARNING": 1, "DANGER": 2, "CRITICAL": 3}

# A single short beep is easy to sleep through, so DANGER/CRITICAL keep
# repeating until the driver recovers. WARNING still beeps only once.
BEEP_REPEAT_SEC = {"DANGER": 2.0, "CRITICAL": 1.0}
MIN_BEEP_GAP_SEC = 1.0   # never beep more often than this, even if the status flickers
VOICE_REPEAT_SEC = 6.0   # spoken warning repeats this often while CRITICAL
EPISODE_END_SEC = 3.0    # calm this long before a new alert is logged again
VOICE_WARNING = "Warning. Critical driver impairment detected. Please pull over."

# Yawn: inner-lip MAR is about 0.00-0.08 with the mouth closed, 0.19-0.26
# while talking and about 0.38 in a yawn (measured on real face photos).
# The mouth must also stay that open for a while — talking opens it for
# well under a second at a time, a yawn holds it open for several seconds.
YAWN_MIN_MAR = 0.32
YAWN_ABOVE_BASELINE = 0.25   # for drivers whose mouth rests slightly open
YAWN_MIN_SEC = 1.5
# Eyes squint during a yawn (EAR drops to about 0.13-0.17), which would look
# like closing eyes. Eye closure is ignored while the mouth is yawning-wide,
# but only for this long, so sleeping with the jaw dropped open is still caught.
YAWN_EYE_IGNORE_SEC = 6.0

# YOLO often misses a held phone for a frame or two, so the phone timer only
# resets after the phone has been gone this long.
PHONE_GRACE_SEC = 1.0

_voice_thread = None


def speak_async(message):
    """Fire voice alert on a background thread so it never blocks the camera loop."""
    global _voice_thread
    if _voice_thread is not None and _voice_thread.is_alive():
        return  # already speaking, don't stack requests
    _voice_thread = threading.Thread(target=speak_warning, args=(message,), daemon=True)
    _voice_thread.start()


_emotion_thread = None
_emotion_result = {"emotion": None, "seq": 0}  # seq goes up each time a reading finishes


def emotion_async(frame):
    """Run DeepFace on a background thread so the camera loop never waits for it.
    Skips the request if the previous reading is still running."""
    global _emotion_thread
    if _emotion_thread is not None and _emotion_thread.is_alive():
        return

    def work():
        _emotion_result["emotion"] = detect_emotion(frame)["emotion"]
        _emotion_result["seq"] += 1

    _emotion_thread = threading.Thread(target=work, args=(), daemon=True)
    _emotion_thread.start()


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
    cause = f" ({state['cause']})" if state.get("cause") else ""
    cv2.putText(frame, f"Status: {state['status']}{cause}  Score: {state['score']}", (10, y),
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
    mouth_open_since = None  # time the mouth first opened yawn-wide; None otherwise
    eyes_closed_since = None  # time the eyes first closed; None while they're open
    current_emotion = "Neutral"
    prev_status = "SAFE"
    last_emotion_at = float("-inf")
    head_baseline = {"pitch": profile["pitch_mean"], "yaw": profile["yaw_mean"]}
    no_face_since = None  # time the face was first lost; None while it's visible
    away_since = None     # time the head first turned away; None while facing the road
    phone_since = None    # time a phone was first seen; None when no phone
    phone_last_seen = None
    last_beep_at = float("-inf")
    last_voice_at = float("-inf")
    episode_peak = None   # worst status logged in the current alert episode; None when calm
    calm_since = None     # when the driver last dropped out of DANGER/CRITICAL
    prev_time = time.time()
    fps = 0

    print("Co-Pilot active. Press 'q' to stop driving.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera disconnected.")
            break

        # ==========================================
        # 1. RUN THE AI MODELS
        # ==========================================
        landmarks = get_landmarks(frame)

        if landmarks:
            no_face_since = None
            left = compute_ear(landmarks, LEFT_EYE)
            right = compute_ear(landmarks, RIGHT_EYE)
            current_ear = (left + right) / 2.0
            current_mar = compute_mar(landmarks)
            # angles relative to this driver's calibrated straight-ahead position
            pose_result = get_pose(landmarks, frame.shape, baseline=head_baseline)
        else:
            # No face is NOT proof the driver is fine — a head dropping forward
            # in sleep, a hand over the face or a covered camera all look like
            # this. Leave the readings empty and time how long it lasts instead.
            if no_face_since is None:
                no_face_since = time.time()
            current_ear = None
            current_mar = None
            pose_result = None
        no_face_sec = time.time() - no_face_since if no_face_since is not None else 0.0

        # Emotion: started every EMOTION_EVERY_SEC seconds in the background;
        # the latest finished reading is picked up here.
        if time.time() - last_emotion_at >= EMOTION_EVERY_SEC:
            emotion_async(frame.copy())
            last_emotion_at = time.time()
        if _emotion_result["emotion"]:
            current_emotion = _emotion_result["emotion"]

        # Phone detection: no real object-detection model wired in yet —
        # kept as an always-empty placeholder on purpose (tracked separately).
        # get_detections already keeps only "cell phone" boxes at confidence 0.5+.
        # Timed in seconds, with a short grace period for missed frames.
        raw_detections = get_detections(frame)
        if raw_detections:
            phone_last_seen = time.time()
            if phone_since is None:
                phone_since = phone_last_seen
        elif phone_since is not None and time.time() - phone_last_seen > PHONE_GRACE_SEC:
            phone_since = None
        # first sighting -> latest sighting, so the grace period itself doesn't add time
        phone_sec = phone_last_seen - phone_since if phone_since is not None else 0.0
        phone_result = {"seen_sec": phone_sec, "phone_detected": phone_sec >= PHONE_WARNING_SEC}

        # Looking away, timed in seconds. With no face the timer keeps running
        # (a head turned far enough loses the face) and the face timer adds
        # its own points.
        if pose_result is not None:
            if pose_result["looking_away"]:
                if away_since is None:
                    away_since = time.time()
            else:
                away_since = None
        away_sec = time.time() - away_since if away_since is not None else 0.0
        head_result = {"away_sec": away_sec}

        # ==========================================
        # 2. PERSONALIZED EYE / YAWN CHECKS (against calibrated baseline)
        # ==========================================
        # Mouth first: a yawn counts only once the mouth has stayed wide open
        # for YAWN_MIN_SEC, so talking and laughing don't count.
        yawn_mar = max(YAWN_MIN_MAR, profile['mar_mean'] + YAWN_ABOVE_BASELINE)
        if current_mar is not None and current_mar > yawn_mar:
            if mouth_open_since is None:
                mouth_open_since = time.time()
        elif current_mar is not None:
            mouth_open_since = None
        mouth_open_sec = time.time() - mouth_open_since if mouth_open_since is not None else 0.0
        yawn_now = mouth_open_sec >= YAWN_MIN_SEC
        if yawn_now and not is_yawning:
            total_yawns += 1
        is_yawning = yawn_now
        yawn_result = {"mar": current_mar, "yawning": is_yawning}

        # Eyes are timed in real seconds (not frames) so the limits mean the
        # same on a fast or slow computer. With no face the timer keeps
        # running, so eyes closing and then the head dropping out of view
        # doesn't wipe the evidence.
        squinting_from_yawn = mouth_open_since is not None and mouth_open_sec < YAWN_EYE_IGNORE_SEC
        if current_ear is not None:
            if current_ear < (profile['ear_mean'] * 0.8) and not squinting_from_yawn:
                if eyes_closed_since is None:
                    eyes_closed_since = time.time()
            else:
                eyes_closed_since = None
        eyes_closed_sec = time.time() - eyes_closed_since if eyes_closed_since is not None else 0.0
        eye_result = {"ear": current_ear, "closed_sec": eyes_closed_sec}

        emotion_result = {"emotion": current_emotion}
        face_result = {"missing_sec": no_face_sec}

        # ==========================================
        # 3. FUSION — single source of truth for the alert decision
        # ==========================================
        state = calculate_driver_state(eye_result, yawn_result, phone_result, head_result,
                                       emotion_result, face_result)

        # ==========================================
        # 4. ALERTS
        # ==========================================
        now = time.time()
        status = state["status"]
        level = STATUS_TO_BEEP_LEVEL.get(status)
        got_worse = STATUS_RANK[status] > STATUS_RANK[prev_status]

        # Beep right away when things get worse, then keep repeating for as
        # long as DANGER/CRITICAL lasts.
        repeat_sec = BEEP_REPEAT_SEC.get(status)
        since_beep = now - last_beep_at
        if level and since_beep >= MIN_BEEP_GAP_SEC and (
                got_worse or (repeat_sec is not None and since_beep >= repeat_sec)):
            play_beep(level)
            last_beep_at = now

        # Spoken warning repeats while CRITICAL lasts
        if status == "CRITICAL" and now - last_voice_at >= VOICE_REPEAT_SEC:
            speak_async(VOICE_WARNING)
            last_voice_at = now

        # SOS: unchanged — fires on every entry into CRITICAL (to be reworked
        # together with the SOS feature).
        if status == "CRITICAL" and prev_status != "CRITICAL":
            sos_async(driver_name)

        # History: one row when an alert episode starts, one more if it
        # escalates to CRITICAL. Flickering in and out doesn't add rows; the
        # episode only ends after EPISODE_END_SEC of calm.
        if state["alarm"]:
            calm_since = None
            if episode_peak is None or (status == "CRITICAL" and episode_peak != "CRITICAL"):
                total_alerts += 1
                log_event(session_id, level, ear=current_ear, mar=current_mar,
                          yaw=pose_result["yaw"] if pose_result else None,
                          phone=phone_result["phone_detected"], emotion=current_emotion)
                episode_peak = status
        elif episode_peak is not None:
            if calm_since is None:
                calm_since = now
            if now - calm_since >= EPISODE_END_SEC:
                episode_peak = None

        prev_status = status

        # ==========================================
        # 5. VIDEO + HUD
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

    profile = load_profile(driver_name)  # None if missing, outdated or incomplete

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
                result = run_calibration(driver_name, cap, duration_sec=5, window_name=WINDOW_NAME)
                if result.get("calibrated"):
                    profile = result
                    print("Calibration complete.")
                else:
                    # keep the previous profile (if any); a bad recording is never used
                    print(f"Calibration failed: {result['error']}. Press 'c' to try again.")
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
