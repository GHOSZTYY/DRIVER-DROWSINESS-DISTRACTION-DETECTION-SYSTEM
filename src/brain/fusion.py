"""
Fusion Engine

This module combines the outputs from all detection modules
(Eye, Yawn, Phone and Head Pose) to calculate a single
Driver Safety Score and determine the driver's state.
"""

def calculate_driver_state(
    eye_result,
    yawn_result,
    phone_result,
    head_result
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
    if eye_result and eye_result.get("drowsy", False):
        score += 4
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
        "reasons": reasons
    }
