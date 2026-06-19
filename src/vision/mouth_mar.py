import numpy as np


def compute_mar(landmarks):
    """
    Compute Mouth Aspect Ratio (MAR)

    Mouth landmarks:
    13  -> upper lip
    14  -> lower lip
    81  -> upper mouth
    311 -> lower mouth
    61  -> left mouth corner
    291 -> right mouth corner
    """

    p13 = np.array(landmarks[13])
    p14 = np.array(landmarks[14])

    p81 = np.array(landmarks[81])
    p311 = np.array(landmarks[311])

    p61 = np.array(landmarks[61])
    p291 = np.array(landmarks[291])

    vertical_1 = np.linalg.norm(p13 - p14)
    vertical_2 = np.linalg.norm(p81 - p311)

    horizontal = np.linalg.norm(p61 - p291)

    if horizontal == 0:
        return 0.0

    mar = (vertical_1 + vertical_2) / (2 * horizontal)

    return float(mar)


def detect_yawn(mar, threshold=0.6):
    """
    Detect if the driver is yawning
    """

    yawning = mar > threshold

    return {
        "mar": round(mar, 3),
        "yawning": yawning
    }