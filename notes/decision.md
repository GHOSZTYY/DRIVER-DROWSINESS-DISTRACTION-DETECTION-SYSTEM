
### get_landmarks(frame) — src/vision/face_mesh.py — Mridul
- **Input:** a webcam frame (NumPy array, BGR format) "The format we have to use to get the webcam frame"
- **Output:** `dict { index: (x_px, y_px, z_norm) }` for all 468 points "we will get output like this pointers" 
- **Returns `None` if no face detected — all callers must handle None!**

The thresholds:- 
#### Mouth landmark indexes (for Samriddhi's mouth_mar.py):
- Top lip: 13
- Bottom lip: 14
- Left corner: 61
- Right corner: 291
- Top mid: 81
- Bottom mid: 311

#### Eye landmark indexes (for eye_ear.py):
- Left eye: 362, 385, 387, 263, 373, 380
- Right eye: 33, 160, 158, 133, 153, 144

#### Head pose landmarks (for head_pose.py):
- Nose tip: 1
- Chin: 152
- Left eye corner: 263
- Right eye corner: 33
- Left mouth: 287
- Right mouth: 57
