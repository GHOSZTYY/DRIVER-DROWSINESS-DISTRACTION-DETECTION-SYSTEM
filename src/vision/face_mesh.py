import cv2
import mediapipe as mp

# Load MediaPipe FaceMesh once when this file is imported
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,        # We only care about the driver's face
    refine_landmarks=True,  # Gives more accurate eye and mouth points
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_landmarks(frame):
    """
    Detect face and return 468 landmark points.
    
    Input:  frame — a webcam image (NumPy array, BGR format)
    Output: dict { index: (x_pixels, y_pixels, z_norm) } for all 468 points
            OR None if no face is detected
    """
    # MediaPipe needs RGB, but webcam gives BGR — so we convert
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(rgb_frame)
    
    # If no face found, return None — callers must handle this!
    if not results.multi_face_landmarks:
        return None
    
    # Get frame dimensions to convert normalised coords to pixels
    h, w = frame.shape[:2]
    
    # Extract all 468 landmarks as pixel coordinates
    landmarks = results.multi_face_landmarks[0].landmark
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
    from webcam import get_frame, cap
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