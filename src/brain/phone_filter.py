"""
Phone filter

YOLO says "there is a phone in the picture". This module decides whether
that phone is the driver USING a phone:

- Mounted phone: a phone in a dashboard/windscreen mount doesn't move. A
  phone held in a hand always moves a little. A phone whose box stays
  within MOUNTED_MAX_MOVE of the same spot for MOUNTED_STILL_SEC is marked
  as mounted, and that spot is ignored for the rest of the drive (unless
  the phone is picked up and moves away).
- Passenger's phone: only phones horizontally within PHONE_REACH face
  widths of the driver's face are counted. A passenger sits further to
  the side.
"""

import math

MOUNTED_STILL_SEC = 4.0     # still this long -> mounted (before the 5 s DANGER limit)
MOUNTED_MAX_MOVE = 0.02     # max wobble, as a fraction of the frame width (13 px at 640)
PHONE_REACH = 1.0           # face widths either side of the face the driver's hands reach

# MediaPipe face-edge landmarks (left and right side of the face at cheek level)
_FACE_LEFT, _FACE_RIGHT = 234, 454


def _center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def near_driver(box, landmarks):
    """True if the phone box is within reach of the driver's face.
    With no face visible every phone counts (can't tell, so be safe)."""
    if not landmarks:
        return True
    left = min(landmarks[_FACE_LEFT][0], landmarks[_FACE_RIGHT][0])
    right = max(landmarks[_FACE_LEFT][0], landmarks[_FACE_RIGHT][0])
    width = max(right - left, 1)
    cx, _ = _center(box)
    return left - PHONE_REACH * width <= cx <= right + PHONE_REACH * width


class PhoneFilter:
    def __init__(self):
        self.track = []          # (time, cx, cy) of the counted phone, oldest first
        self.mounted_at = None   # (cx, cy) of a phone identified as mounted

    def reset(self):
        self.track = []
        self.mounted_at = None

    def update(self, detections, landmarks, frame_width, now):
        """Returns the detection that counts as the driver using a phone,
        or None. `detections` are dicts with "confidence" and "box"."""
        max_move = MOUNTED_MAX_MOVE * frame_width

        phones = [d for d in detections if near_driver(d["box"], landmarks)]
        if not phones:
            return None
        phone = max(phones, key=lambda d: d["confidence"])
        cx, cy = _center(phone["box"])

        # the known mounted phone, still in its mount
        if self.mounted_at is not None and math.dist((cx, cy), self.mounted_at) <= max_move:
            return None

        # forget positions older than the stillness window
        self.track.append((now, cx, cy))
        span_start = now - MOUNTED_STILL_SEC
        while len(self.track) > 1 and self.track[1][0] <= span_start:
            self.track.pop(0)

        if now - self.track[0][0] >= MOUNTED_STILL_SEC:
            mx = sum(p[1] for p in self.track) / len(self.track)
            my = sum(p[2] for p in self.track) / len(self.track)
            if all(math.dist((p[1], p[2]), (mx, my)) <= max_move for p in self.track):
                self.mounted_at = (mx, my)
                self.track = []
                return None

        return phone

    def lost(self):
        """Call when the phone has been gone longer than the grace period."""
        self.track = []
