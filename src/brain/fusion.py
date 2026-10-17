"""
Fusion Engine

This module combines the outputs from all detection modules
(Eye, Yawn, Phone and Head Pose) to calculate a single
Driver Safety Score and determine the driver's state.
"""

# Eyes closed: normal blinks (0.1-0.4s) and slow tired blinks (under ~1s) are
# ignored; staying closed longer is treated as falling asleep. Shorter than the
# face limits because closed eyes are stronger proof of sleep than a lost face.
EYES_CLOSED_DROWSY_SEC = 1.5
EYES_CLOSED_CRITICAL_SEC = 3.0

# Face not visible: short gaps (blinks, a quick shoulder check, a detection
# glitch) are ignored; longer gaps are treated as a head drop or a blocked camera.
NO_FACE_DROWSY_SEC = 2.0
NO_FACE_CRITICAL_SEC = 5.0


def calculate_driver_state(
    eye_result,
    yawn_result,
    phone_result,
    head_result,
    emotion_result,
    face_result=None
):
    """
    Combines outputs from all detectors and calculates
    the driver's safety score.

    Returns:
        {
            "score": int,
            "status": str,
            "alarm": bool,
            "reasons": list
        }
    """

    score = 0
    reasons = []

    # Eye Detection
    if eye_result:
        closed_sec = eye_result.get("closed_sec", 0)

        if closed_sec >= EYES_CLOSED_CRITICAL_SEC:
            score += 8
            reasons.append("Eyes closed 3s+")

        elif closed_sec >= EYES_CLOSED_DROWSY_SEC:
            score += 5
            reasons.append("Eyes closed")

    # Yawn Detection
    if yawn_result and yawn_result.get("yawning", False):
        score += 2
        reasons.append("Yawning")

    # Phone Detection
    if phone_result and phone_result.get("phone_detected", False):
        score += 3
        reasons.append("Phone detected")

    # Head Pose Detection
    if head_result and head_result.get("distracted", False):
        score += 2
        reasons.append("Looking away")

    # Face Not Visible
    if face_result:
        missing_sec = face_result.get("missing_sec", 0)

        if missing_sec >= NO_FACE_CRITICAL_SEC:
            score += 8
            reasons.append("Face not visible 5s+")

        elif missing_sec >= NO_FACE_DROWSY_SEC:
            score += 5
            reasons.append("Face not visible")

    # Emotion Detection
    if emotion_result:
        emotion = emotion_result.get("emotion")

        if emotion == "angry":
            score += 2
            reasons.append("Angry")

        elif emotion == "fear":
            score += 2
            reasons.append("Fear")

        elif emotion == "sad":
            score += 1
            reasons.append("Sad")

        elif emotion == "surprise":
            score += 1
            reasons.append("Surprised")

    # Decide Driver Status
    if score == 0:
       status = "SAFE"
       alarm = False

    elif score <= 4:
        status = "WARNING"
        alarm = False

    elif score <= 7:
        status = "DROWSY"
        alarm = True

    else:
        status = "CRITICAL"
        alarm = True

    return {
        "score": score,
        "status": status,
        "alarm": alarm,
        "emotion": emotion_result.get("emotion") if emotion_result else None,
        "reasons": reasons
    }
