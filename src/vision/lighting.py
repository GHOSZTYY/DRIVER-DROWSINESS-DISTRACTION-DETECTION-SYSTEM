"""
Lighting checks

A normal webcam has no infrared, so at night or in a dark cabin the face
detector starts missing the face. This module:

- brightens dark frames (CLAHE contrast boost on the lightness channel)
  before they go to the face detector, which recovers detection in dim
  light, and
- says when the picture is too dark to trust, and
- spots eyes that are covered (sunglasses): the eye areas come out much
  darker than the cheeks.

None of this replaces an infrared camera for real night driving.
"""

import cv2
import numpy as np

LOW_LIGHT_LEVEL = 80        # mean brightness (0-255) below which frames are boosted
TOO_DARK_LEVEL = 25         # below this even boosted frames are unreliable
EYES_COVERED_RATIO = 0.45   # eye areas darker than this fraction of the cheeks = covered

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# Landmark indexes: around each eye, and a patch on each cheek
_LEFT_EYE_AREA = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE_AREA = [33, 160, 158, 133, 153, 144]
_CHEEKS = [50, 280]


def brightness(frame):
    return float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())


def enhance_if_dark(frame):
    """Returns (frame_for_detection, brightness). Dark frames are boosted;
    bright enough frames are returned unchanged."""
    level = brightness(frame)
    if level >= LOW_LIGHT_LEVEL:
        return frame, level
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    boosted = cv2.merge((_clahe.apply(l_channel), a, b))
    return cv2.cvtColor(boosted, cv2.COLOR_LAB2BGR), level


def detect_face(frame, detector):
    """Run the face detector on the normal frame first. Only if that finds no
    face and the frame is dark, try again on a brightened copy. (Tested on
    darkened, noisy photos: never worse than the plain frame, and finds more
    faces in very dark frames; boosting every dark frame was slightly worse
    in medium darkness.) Returns (landmarks or None, brightness)."""
    landmarks = detector(frame)
    level = brightness(frame)
    if not landmarks and level < LOW_LIGHT_LEVEL:
        landmarks = detector(enhance_if_dark(frame)[0])
    return landmarks, level


def light_notice(level):
    """Text to show on screen about the lighting, or None if it's fine."""
    if level < TOO_DARK_LEVEL:
        return "TOO DARK - monitoring unreliable"
    if level < LOW_LIGHT_LEVEL:
        return "Low light - detection less reliable"
    return None


def _mean_in_box(gray, points, pad):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    h, w = gray.shape
    x1, x2 = max(min(xs) - pad, 0), min(max(xs) + pad, w)
    y1, y2 = max(min(ys) - pad, 0), min(max(ys) + pad, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return float(gray[y1:y2, x1:x2].mean())


def eyes_covered(frame, landmarks):
    """True if both eye areas are much darker than the cheeks (sunglasses).
    Returns False if it can't tell."""
    if not landmarks:
        return False
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_w = abs(landmarks[454][0] - landmarks[234][0]) if 454 in landmarks and 234 in landmarks else 100
    pad = max(int(face_w * 0.03), 2)
    eyes = [_mean_in_box(gray, [landmarks[i][:2] for i in area], pad)
            for area in (_LEFT_EYE_AREA, _RIGHT_EYE_AREA)]
    cheeks = [_mean_in_box(gray, [landmarks[i][:2]], int(face_w * 0.06) or 3) for i in _CHEEKS]
    if None in eyes or None in cheeks or np.mean(cheeks) < 1:
        return False
    return max(eyes) / float(np.mean(cheeks)) < EYES_COVERED_RATIO
