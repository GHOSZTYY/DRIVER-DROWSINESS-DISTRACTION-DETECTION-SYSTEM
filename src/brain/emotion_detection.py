"""
Emotion Detection Module

Uses DeepFace to detect the driver's dominant emotion.
"""

from deepface import DeepFace


def detect_emotion(frame):
    """
    Detects the dominant emotion from a video frame.

    Args:
        frame: OpenCV image (BGR)

    Returns:
        {
            "emotion": str | None
        }
    """

    try:
        result = DeepFace.analyze(
        img_path=frame,
        actions=["emotion"],
        detector_backend="opencv",
        enforce_detection=False,
        silent=True
    )

        # DeepFace may return a list or a dictionary
        if isinstance(result, list):
            result = result[0]

        emotion = result.get("dominant_emotion")

        return {
            "emotion": emotion
        }

    except Exception:
        return {
            "emotion": None
        }