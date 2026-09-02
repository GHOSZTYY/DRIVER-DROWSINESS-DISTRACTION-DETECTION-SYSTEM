import cv2

# Open the webcam once and keep it open
# CAP_DSHOW avoids the multi-second freeze/black-frame issue on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: Could not open webcam (index 0). Is it in use by another app?")

def get_frame():
    """Read a single frame from the webcam. Returns the frame as a NumPy array."""
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def show_live():
    """Show live webcam feed in a window. Press Q to quit."""
    print("Camera starting... Press Q to quit.")
    while True:
        frame = get_frame()
        if frame is None:
            print("Error: Could not read from camera.")
            break
        cv2.imshow("Driver Safety - Webcam Test", frame)
        # Wait 1ms for a key press. If Q is pressed, exit.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("Camera closed.")

# If you run this file directly, show the live feed
if __name__ == "__main__":
    show_live()