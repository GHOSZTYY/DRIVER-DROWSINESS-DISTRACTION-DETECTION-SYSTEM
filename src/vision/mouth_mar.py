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
if __name__ == "__main__":
    import cv2
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.vision.webcam import get_frame, cap
    from src.vision.face_mesh import get_landmarks

    print("MAR detector running... Press Q to quit.")
    print("Open your mouth wide to simulate a yawn!")

    while True:
        frame = get_frame()
        if frame is None:
            continue

        landmarks = get_landmarks(frame)

        if landmarks:
            mar = compute_mar(landmarks)
            result = detect_yawn(mar)
            color = (0, 0, 255) if result["yawning"] else (0, 255, 0)
            label = "YAWNING!" if result["yawning"] else "Normal"
            cv2.putText(frame, f"MAR: {result['mar']}  |  {label}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)
        else:
            cv2.putText(frame, "No face detected",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)

        cv2.imshow("MAR Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()