import numpy as np


def compute_mar(landmarks):
    """
    Compute Mouth Aspect Ratio (MAR) from the inner lip edge.

    Three vertical pairs (upper inner lip -> lower inner lip):
    13  -> 14   centre
    81  -> 178  left of centre
    311 -> 402  right of centre
    One horizontal pair (outer corners):
    61  -> 291

    81 and 311 are both on the UPPER inner lip, so the old 81 -> 311 pair
    measured across the lip, not its opening (about 0.24 with the mouth
    closed). With matching upper/lower pairs a closed mouth reads about 0.
    """

    def point(i):
        return np.array(landmarks[i][:2])

    vertical_1 = np.linalg.norm(point(13) - point(14))
    vertical_2 = np.linalg.norm(point(81) - point(178))
    vertical_3 = np.linalg.norm(point(311) - point(402))

    horizontal = np.linalg.norm(point(61) - point(291))

    if horizontal == 0:
        return 0.0

    mar = (vertical_1 + vertical_2 + vertical_3) / (3 * horizontal)

    return float(mar)


def detect_yawn(mar, threshold=0.32):
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