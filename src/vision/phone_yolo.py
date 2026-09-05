import os
from ultralytics import YOLO

# COCO class id 67 = "cell phone" — the only class we care about here.
_PHONE_CLASS_ID = 67

_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'yolov8n.pt'))
os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)

# Auto-downloads to _MODEL_PATH on first run if not already there (same pattern as face_mesh.py's model).
_model = YOLO(_MODEL_PATH)


def get_detections(frame, confidence_threshold=0.5):
    """
    Run YOLO object detection on a frame, restricted to the "cell phone" class.

    Input:  frame — webcam image (NumPy array, BGR)
    Output: list of { "label": str, "confidence": float } — shape expected by
            PhoneDetector.detect() in src/brain/phone_detector.py
    """
    results = _model.predict(
        frame,
        conf=confidence_threshold,
        classes=[_PHONE_CLASS_ID],
        verbose=False,
    )

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "label": _model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
            })
    return detections


# Test: run this file directly to see phone detection live
if __name__ == "__main__":
    import sys
    import cv2
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.vision.webcam import get_frame, cap
    from src.brain.phone_detector import PhoneDetector

    tracker = PhoneDetector(confidence_threshold=0.4, frame_threshold=10)

    print("Phone detector running... Press Q to quit.")
    while True:
        frame = get_frame()
        if frame is None:
            continue

        detections = get_detections(frame)
        result = tracker.detect(detections)

        label = "PHONE DETECTED!" if result["phone_detected"] else "No phone"
        color = (0, 0, 255) if result["phone_detected"] else (0, 255, 0)
        cv2.putText(frame, f"{label}  conf:{result['confidence']}  frames:{result['phone_frames']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        cv2.imshow("Phone Detection Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
