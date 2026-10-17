import cv2
import os
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# MediaPipe 1.0 dropped the legacy `mp.solutions.face_mesh` API in favor of
# the Tasks API (FaceLandmarker), which needs a model file downloaded once.
_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'face_landmarker.task'))


def _ensure_model():
    if not os.path.exists(_MODEL_PATH):
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        print("Downloading face_landmarker.task model (first run only)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)


_ensure_model()

_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
    running_mode=mp_vision.RunningMode.IMAGE,  # each frame processed independently, no video timestamps needed
    # Look for up to 3 faces and pick the driver ourselves (see _pick_driver);
    # with num_faces=1 MediaPipe could return a passenger's face instead.
    num_faces=3,
)
_landmarker = mp_vision.FaceLandmarker.create_from_options(_options)

# Face-edge landmarks (left/right cheek) used to measure face size
_FACE_LEFT, _FACE_RIGHT = 234, 454
# Keep following the same face if it's within this fraction of the frame
# width of where the driver was last seen
FOLLOW_DISTANCE = 0.25

_last_driver_x = None  # normalised x of the driver's face centre last frame


def _pick_driver(faces):
    """The driver sits closest to the camera, so their face is the biggest.
    If the face seen last frame is still near the same place, keep following
    it, so a passenger leaning in for a moment can't take over."""
    global _last_driver_x

    def width(face):
        return abs(face[_FACE_RIGHT].x - face[_FACE_LEFT].x)

    def centre(face):
        return (face[_FACE_RIGHT].x + face[_FACE_LEFT].x) / 2

    chosen = None
    if _last_driver_x is not None:
        near = [f for f in faces if abs(centre(f) - _last_driver_x) <= FOLLOW_DISTANCE]
        if near:
            chosen = max(near, key=width)
    if chosen is None:
        chosen = max(faces, key=width)
    _last_driver_x = centre(chosen)
    return chosen


def reset():
    """Forget which face was the driver — call when a new drive starts."""
    global _last_driver_x
    _last_driver_x = None


def get_landmarks(frame):
    """
    Detect face and return landmark points (478 points, incl. iris — a superset
    of the old 468-point legacy Face Mesh, same index numbering for points 0-467).

    Input:  frame — a webcam image (NumPy array, BGR format)
    Output: dict { index: (x_pixels, y_pixels, z_norm) } for all landmarks
            OR None if no face is detected
    """
    # MediaPipe needs RGB, but webcam gives BGR — so we convert
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = _landmarker.detect(mp_image)

    # If no face found, return None — callers must handle this!
    if not result.face_landmarks:
        return None

    # Get frame dimensions to convert normalised coords to pixels
    h, w = frame.shape[:2]

    landmarks = _pick_driver(result.face_landmarks)
    return {
        i: (int(lm.x * w), int(lm.y * h), lm.z)
        for i, lm in enumerate(landmarks)
    }


def draw_landmarks(frame, landmarks):
    """
    Draw all 468 face points as green dots on the frame.
    Only used for testing — not needed in the final app.
    
    Input:  frame — webcam image
            landmarks — dict from get_landmarks()
    Output: frame with dots drawn on it
    """
    if landmarks is None:
        return frame
    for i, (x, y, z) in landmarks.items():
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
    return frame


# Test: run this file directly to see landmarks on your face
if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.vision.webcam import get_frame, cap
    import time

    print("Face mesh running... Press Q to quit.")
    while True:
        frame = get_frame()
        if frame is None:
            continue

        t1 = time.perf_counter()
        landmarks = get_landmarks(frame)
        t2 = time.perf_counter()

        if landmarks:
            frame = draw_landmarks(frame, landmarks)
            ms = (t2 - t1) * 1000
            cv2.putText(frame, f"Landmarks: 468  |  {ms:.1f}ms",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow("Face Mesh Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")