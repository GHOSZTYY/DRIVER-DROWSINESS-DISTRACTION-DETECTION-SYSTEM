"""
Fusion Engine

This module combines the outputs from all detection modules
(Eye, Yawn, Phone and Head Pose) to calculate a single
Driver Safety Score and determine the driver's state.
"""

# Score bands: 0 SAFE, 1-4 WARNING, 5-7 DANGER, 8+ CRITICAL
DANGER_SCORE = 5
CRITICAL_SCORE = 8

# Eyes closed: normal blinks (0.1-0.4s) and slow tired blinks (under ~1s) are
# ignored; staying closed longer is treated as falling asleep. Shorter than the
# face limits because closed eyes are stronger proof of sleep than a lost face.
EYES_CLOSED_DANGER_SEC = 1.5
EYES_CLOSED_CRITICAL_SEC = 3.0

# Face not visible: short gaps (blinks, a quick shoulder check, a detection
# glitch) are ignored; longer gaps are treated as a head drop or a blocked camera.
NO_FACE_DANGER_SEC = 2.0
NO_FACE_CRITICAL_SEC = 5.0

# Looking away: mirror and shoulder checks take under ~1.5s. Eyes off the road
# for more than 2s roughly doubles crash risk (100-Car Naturalistic Driving
# Study, 2006), so 2s raises DANGER and twice that raises CRITICAL.
LOOKING_AWAY_DANGER_SEC = 2.0
LOOKING_AWAY_CRITICAL_SEC = 4.0

# Phone in view: under 1s is treated as a detection glitch. Holding it for 5s+
# is real phone use. Phone alone never reaches CRITICAL — looking down at it
# adds the looking-away points on top.
PHONE_WARNING_SEC = 1.0
PHONE_DANGER_SEC = 5.0


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

    Inputs (any can be None):
        eye_result     {"closed_sec": float}
        yawn_result    {"yawning": bool}
        phone_result   {"seen_sec": float}
        head_result    {"away_sec": float}
        emotion_result {"emotion": str}
        face_result    {"missing_sec": float}

    Returns:
        {
            "score": int,
            "status": str,        # SAFE / WARNING / DANGER / CRITICAL
            "alarm": bool,
            "cause": str | None,  # Drowsiness / Distraction / Drowsiness + Distraction
            "reasons": list
        }
    """

    score = 0
    reasons = []
    drowsy_sign = False
    distracted_sign = False

    # Eye Detection
    if eye_result:
        closed_sec = eye_result.get("closed_sec", 0)

        if closed_sec >= EYES_CLOSED_CRITICAL_SEC:
            score += 8
            reasons.append("Eyes closed 3s+")
            drowsy_sign = True

        elif closed_sec >= EYES_CLOSED_DANGER_SEC:
            score += 5
            reasons.append("Eyes closed")
            drowsy_sign = True

    # Yawn Detection
    if yawn_result and yawn_result.get("yawning", False):
        score += 2
        reasons.append("Yawning")
        drowsy_sign = True

    # Phone Detection
    if phone_result:
        seen_sec = phone_result.get("seen_sec", 0)

        if seen_sec >= PHONE_DANGER_SEC:
            score += 5
            reasons.append("Phone in use 5s+")
            distracted_sign = True

        elif seen_sec >= PHONE_WARNING_SEC:
            score += 3
            reasons.append("Phone detected")
            distracted_sign = True

    # Head Pose Detection
    if head_result:
        away_sec = head_result.get("away_sec", 0)

        if away_sec >= LOOKING_AWAY_CRITICAL_SEC:
            score += 8
            reasons.append("Looking away 4s+")
            distracted_sign = True

        elif away_sec >= LOOKING_AWAY_DANGER_SEC:
            score += 5
            reasons.append("Looking away")
            distracted_sign = True

    # Face Not Visible
    if face_result:
        missing_sec = face_result.get("missing_sec", 0)

        if missing_sec >= NO_FACE_CRITICAL_SEC:
            score += 8
            reasons.append("Face not visible 5s+")
            drowsy_sign = True

        elif missing_sec >= NO_FACE_DANGER_SEC:
            score += 5
            reasons.append("Face not visible")
            drowsy_sign = True

    # Emotion Detection
    emotion_points = 0
    if emotion_result:
        emotion = emotion_result.get("emotion")

        if emotion == "angry":
            emotion_points = 2
            reasons.append("Angry")

        elif emotion == "fear":
            emotion_points = 2
            reasons.append("Fear")

        elif emotion == "sad":
            emotion_points = 1
            reasons.append("Sad")

        elif emotion == "surprise":
            emotion_points = 1
            reasons.append("Surprised")

    # Emotion guesses from a single webcam frame are unreliable, so they can
    # add to an alarm but never be the thing that starts one.
    if score < DANGER_SCORE:
        score = min(score + emotion_points, DANGER_SCORE - 1)
    else:
        score += emotion_points

    # Decide Driver Status
    if score == 0:
        status = "SAFE"
        alarm = False

    elif score < DANGER_SCORE:
        status = "WARNING"
        alarm = False

    elif score < CRITICAL_SCORE:
        status = "DANGER"
        alarm = True

    else:
        status = "CRITICAL"
        alarm = True

    # Say what kind of problem it is, so phone use isn't reported as "drowsy"
    if drowsy_sign and distracted_sign:
        cause = "Drowsiness + Distraction"
    elif drowsy_sign:
        cause = "Drowsiness"
    elif distracted_sign:
        cause = "Distraction"
    else:
        cause = None

    return {
        "score": score,
        "status": status,
        "alarm": alarm,
        "cause": cause,
        "emotion": emotion_result.get("emotion") if emotion_result else None,
        "reasons": reasons
    }
