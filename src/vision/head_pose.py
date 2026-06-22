import cv2
import numpy as np

# 3D model points of a generic face (in centimetres)
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # Nose tip        — landmark 1
    (0.0,  -63.6,  -12.5),   # Chin            — landmark 152
    (-43.3,  32.7, -26.0),   # Left eye corner  — landmark 263
    (43.3,   32.7, -26.0),   # Right eye corner — landmark 33
    (-28.9, -28.9, -24.1),   # Left mouth       — landmark 287
    (28.9,  -28.9, -24.1),   # Right mouth      — landmark 57
], dtype=np.float64)

# The 6 landmark indexes that match the 3D model points above
LANDMARK_IDS = [1, 152, 263, 33, 287, 57]

# Thresholds
YAW_THRESHOLD    = 25.0
PITCH_THRESHOLD  = 20.0
DISTRACT_FRAMES  = 45  # ≈1.5 seconds at 30 FPS

# ============================================
# NEW: Camera mounting compensation
# ============================================
# If camera is mounted high (looking down at driver), 
# driver looking straight at road appears "pitched up" to camera.
# Measure this offset once and set it here.
# 
# Examples:
#   CAMERA_PITCH_OFFSET = 0    # Camera at eye level (straight on)
#   CAMERA_PITCH_OFFSET = 25   # Camera mounted 25° above driver
#   CAMERA_PITCH_OFFSET = -15  # Camera mounted below driver (rare)
# ============================================
CAMERA_PITCH_OFFSET = 25.0   # ← ADJUST THIS: degrees camera is tilted down

# Persist between calls
_distract_counter = 0


def get_pose(landmarks, frame_shape):
    """
    Estimate head pose from 6 face landmarks.
    """
    global _distract_counter

    if landmarks is None:
        return None

    h, w = frame_shape[:2]

    # Pull the 6 matching 2D image points
    image_points = np.array([
        landmarks[i][:2] for i in LANDMARK_IDS
    ], dtype=np.float64)

    # Build camera matrix
    focal_length  = w
    center        = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1         ]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        MODEL_POINTS, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Extract Euler angles manually
    sy = np.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        yaw   = np.degrees(np.arctan2(-rmat[2, 0], sy))
        roll  = 0.0

    # ============================================
    # NEW: Compensate for camera mounting angle
    # ============================================
    # Subtract the fixed camera tilt so "looking at road" = pitch ≈ 0
    pitch = pitch - CAMERA_PITCH_OFFSET

    # Check distraction: yaw (left/right) OR pitch (up/down relative to road)
    # After compensation:
    #   pitch ≈ 0  → looking straight at road (normal driving)
    #   pitch > 0  → looking up (ceiling, rearview mirror) → distracted
    #   pitch < 0  → looking down (phone, lap, dashboard) → distracted
    looking_away = abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD

    if looking_away:
        _distract_counter += 1
    else:
        _distract_counter = 0

    distracted = _distract_counter >= DISTRACT_FRAMES

    return {
        "pitch":      round(float(pitch), 2),
        "yaw":        round(float(yaw),   2),
        "roll":       round(float(roll),  2),
        "distracted": distracted
    }


def reset():
    """Reset distraction counter — call when a new session starts."""
    global _distract_counter
    _distract_counter = 0


# Test: run this file directly to see head pose values live
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.vision.webcam import get_frame, cap
    from src.vision.face_mesh import get_landmarks

    print("Head pose running... Press Q to quit.")
    print(f"Camera pitch offset: {CAMERA_PITCH_OFFSET}°")
    print("Look straight at road — pitch should be near 0.")
    print(f"Hold yaw > {YAW_THRESHOLD}° or |pitch| > {PITCH_THRESHOLD}° for 1.5s → DISTRACTED alert.")

    while True:
        frame = get_frame()
        if frame is None:
            continue

        landmarks = get_landmarks(frame)
        result    = get_pose(landmarks, frame.shape)

        if result:
            # Green only for OK, red for everything else
            if result["distracted"]:
                label = "DISTRACTED!"
                color = (0, 0, 255)
            elif abs(result["yaw"]) > YAW_THRESHOLD:
                label = "LOOKING SIDE!"
                color = (0, 0, 255)
            elif abs(result["pitch"]) > PITCH_THRESHOLD:
                label = "LOOKING DOWN/UP!"
                color = (0, 0, 255)
            else:
                label = "OK"
                color = (0, 255, 0)
            
            cv2.putText(frame,
                f"Yaw:{result['yaw']:+.1f}  Pitch:{result['pitch']:+.1f}  Roll:{result['roll']:+.1f}  |  {label}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            # Show calibration info
            cv2.putText(frame,
                f"Camera offset: {CAMERA_PITCH_OFFSET:+.1f}°",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        elif landmarks is None:
            cv2.putText(frame, "NO FACE DETECTED",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            cv2.putText(frame, "FACE DETECTED — CANNOT ESTIMATE POSE",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Head Pose Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()