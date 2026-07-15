class PhoneDetector:
    """
    Phone usage detection module.

    Detects whether a mobile phone is present
    for multiple consecutive frames.
    """

    def __init__(self, confidence_threshold=0.5, frame_threshold=10):
        self.confidence_threshold = confidence_threshold
        self.frame_threshold = frame_threshold

        self.phone_frames = 0
        self.phone_detected = False

    def detect(self, detections):
        """
        detections example:

        [
            {
                "label": "cell phone",
                "confidence": 0.82
            }
        ]
        """

        phone_found = False
        best_confidence = 0.0

        for obj in detections:

            label = obj.get("label")
            confidence = obj.get("confidence", 0)

            if (
                label == "cell phone"
                and confidence >= self.confidence_threshold
            ):
                phone_found = True
                best_confidence = max(
                    best_confidence,
                    confidence
                )

        if phone_found:
            self.phone_frames += 1
        else:
            self.phone_frames = 0

        self.phone_detected = (
            self.phone_frames >= self.frame_threshold
        )

        return {
            "phone_detected": self.phone_detected,
            "confidence": round(best_confidence, 3),
            "phone_frames": self.phone_frames
        }