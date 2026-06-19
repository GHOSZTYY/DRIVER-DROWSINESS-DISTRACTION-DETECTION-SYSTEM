import numpy as np
from scipy.spatial.distance import euclidean

# Landmark indexes for each eye (from MediaPipe face mesh)
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# How many consecutive frames eye must be closed to count as drowsy
# 45 frames ≈ 1.5 seconds at 30 FPS
EAR_THRESHOLD   = 0.25
CLOSED_FRAMES   = 45

# Track how many frames eyes have been closed (persists between calls)
_closed_counter = 0
_blink_count    = 0
_was_closed     = False   # helps detect a completed blink


def compute_ear(landmarks, eye_indexes):
    """
    Compute Eye Aspect Ratio for ONE eye.

    Input:  landmarks   — dict from get_landmarks()
            eye_indexes — list of 6 landmark IDs for that eye
    Output: EAR as a float (lower = more closed)
    """
    # Pull the 6 (x, y) points for this eye
    p = [landmarks[i][:2] for i in eye_indexes]

    # Vertical distances (two pairs)
    v1 = euclidean(p[1], p[5])
    v2 = euclidean(p[2], p[4])

    # Horizontal distance (one pair — eye width)
    h  = euclidean(p[0], p[3])

    # Avoid division by zero
    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return round(ear, 4)


def detect_drowsy(landmarks):
    """
    Check both eyes and decide if the driver is drowsy.

    Input:  landmarks — dict from get_landmarks(), or None
    Output: dict {
                ear:         float  — average EAR of both eyes,
                drowsy:      bool   — True if eyes closed 1.5s+,
                blink_count: int    — total blinks this session
            }
            OR None if landmarks is None
    """
    global _closed_counter, _blink_count, _was_closed

    if landmarks is None:
        return None

    # Compute EAR for each eye then average them
    left_ear  = compute_ear(landmarks, LEFT_EYE)
    right_ear = compute_ear(landmarks, RIGHT_EYE)
    avg_ear   = (left_ear + right_ear) / 2.0

    eyes_closed = avg_ear < EAR_THRESHOLD

    if eyes_closed:
        _closed_counter += 1
        _was_closed = True
    else:
        # Eyes just reopened — count it as a blink if it was brief
        if _was_closed and _closed_counter < CLOSED_FRAMES:
            _blink_count += 1
        _closed_counter = 0
        _was_closed = False

    drowsy = _closed_counter >= CLOSED_FRAMES

    return {
        "ear":         round(avg_ear, 4),
        "drowsy":      drowsy,
        "blink_count": _blink_count
    }


def reset():
    """Reset all counters — call this when a new session starts."""
    global _closed_counter, _blink_count, _was_closed
    _closed_counter = 0
    _blink_count    = 0
    _was_closed     = False


# Test: run this file directly to see EAR values live
if __name__ == "__main__":
    import cv2
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.vision.webcam import get_frame, cap
    from src.vision.face_mesh import get_landmarks

    print("EAR detector running... Press Q to quit.")
    print("Close your eyes for 1.5 seconds to trigger DROWSY alert.")

    while True:
        frame = get_frame()
        if frame is None:
            continue

        landmarks = get_landmarks(frame)
        result    = detect_drowsy(landmarks)

        if result:
            color = (0, 0, 255) if result["drowsy"] else (0, 255, 0)
            label = "DROWSY!" if result["drowsy"] else "Awake"
            cv2.putText(frame, f"EAR: {result['ear']}  |  {label}  |  Blinks: {result['blink_count']}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("EAR Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()