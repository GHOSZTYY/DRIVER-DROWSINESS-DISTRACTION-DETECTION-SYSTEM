# Driver Drowsiness & Distraction Detection System

A real-time, webcam-based driver safety co-pilot. It watches the driver's face
through an ordinary webcam, checks for signs of sleepiness and distraction,
combines everything into one safety score, and raises escalating alerts —
from a soft beep up to a spoken warning and an emergency SMS.

Everything runs locally on a laptop. No special hardware is needed beyond a
webcam, speakers, and (optionally) a Twilio account for SOS messages.

---

## Table of contents

- [What it detects](#what-it-detects)
- [How it works](#how-it-works)
- [Safety score and alert levels](#safety-score-and-alert-levels)
- [Personal calibration](#personal-calibration)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Setup](#setup)
- [Running the app](#running-the-app)
- [Testing individual modules](#testing-individual-modules)
- [Session logging (SQLite)](#session-logging-sqlite)
- [Configuration and tuning](#configuration-and-tuning)
- [Known limitations](#known-limitations)
- [Team](#team)

---

## What it detects

| Signal | How it is measured | Module |
|---|---|---|
| **Eye closure** | Eye Aspect Ratio (EAR) from 6 landmarks per eye, compared with the driver's calibrated baseline | `src/vision/eye_ear.py` |
| **Yawning** | Mouth Aspect Ratio (MAR) from 6 mouth landmarks, compared with the calibrated baseline | `src/vision/mouth_mar.py` |
| **Looking away** | Head yaw / pitch from a 6-point PnP solve (OpenCV `solvePnP`) | `src/vision/head_pose.py` |
| **Phone usage** | YOLOv8n object detection, restricted to the COCO "cell phone" class | `src/vision/phone_yolo.py`, `src/brain/phone_detector.py` |
| **Emotional state** | DeepFace emotion analysis (angry, fear, sad, surprise raise the risk) | `src/brain/emotion_detection.py` |

All face-based signals come from **MediaPipe Face Landmarker** (478 facial
landmarks per frame).

---

## How it works

```
 Webcam frame (640x480)
        │
        ▼
 MediaPipe Face Landmarker ──► EAR (eyes)   ──┐
        │                  ──► MAR (mouth)  ──┤
        │                  ──► Head pose    ──┤
        │                                     │
        ├──► YOLOv8n (cell phone only) ───────┤
        │                                     ▼
        └──► DeepFace (every 15th frame) ─► Fusion engine ──► score + status
                                                               │
                         ┌──────────────┬─────────────────────┼────────────────┐
                         ▼              ▼                     ▼                ▼
                    Beep (pygame)  Voice (pyttsx3)   SOS SMS (Twilio)   SQLite event log
```

1. **Capture** — frames are read from webcam 0 (DirectShow backend on Windows).
2. **Perceive** — each frame goes through the landmark model, the phone
   detector, and (every 15 frames, because it is heavy) the emotion model.
3. **Judge** — eye and yawn readings are checked against the driver's own
   calibrated baseline, and short glitches are filtered out by requiring the
   condition to last several consecutive frames.
4. **Fuse** — `src/brain/fusion.py` adds up weighted points from every signal
   into one score and maps it to a status.
5. **Act** — a beep plays as soon as the status gets worse, and keeps
   repeating while the driver stays `DROWSY` or `CRITICAL`, so a single short
   beep can't be slept through. Voice and SMS run on background threads so
   the video never freezes.
6. **Record** — each session and every alert are written to a local SQLite
   database.

---

## Safety score and alert levels

The fusion engine gives points for each active warning sign:

| Condition | Points |
|---|---|
| Face not visible for 5+ seconds (head dropped, face covered, camera blocked) | +8 |
| Face not visible for 2–5 seconds | +5 |
| Eyes closed for 3+ seconds (EAR below 80% of baseline) | +8 |
| Eyes closed for 1.5–3 seconds | +5 |
| Phone in view (10+ consecutive frames, confidence ≥ 0.5) | +3 |
| Yawning (MAR above 150% of baseline) | +2 |
| Looking away (yaw > 25° or pitch > 20° for 45+ frames) | +2 |
| Emotion: angry or fear | +2 |
| Emotion: sad or surprise | +1 |

The total score sets the driver's status:

| Score | Status | What happens |
|---|---|---|
| 0 | `SAFE` | Nothing — green HUD |
| 1 – 4 | `WARNING` | Soft beep (once) |
| 5 – 7 | `DROWSY` | Medium beep, repeated every 2 s while it lasts; event logged |
| 8+ | `CRITICAL` | Loud beep every 1 s, spoken "please pull over" warning every 6 s, SOS SMS, event logged |

History gets one entry when an alert starts and one more if it gets worse
(`DROWSY` → `CRITICAL`). Flickering in and out of an alert doesn't add
entries; a new alert is only logged after 3 seconds of calm.

The on-screen HUD shows the driver name, FPS, current status and score, the
reasons behind it (for example `Eyes closed, Yawning`), the latest emotion,
and the total number of yawns this session.

---

## Personal calibration

Eye and mouth shapes differ a lot between people, so a fixed EAR/MAR threshold
gives false alarms for some drivers and misses others. Instead, the app records
a **5-second baseline** for each driver while they look ahead normally:

- mean and standard deviation of EAR
- mean MAR

The profile is saved to `data/profiles/<driver_name>.json` and reused on the
next run. If no face is seen during calibration, generic fallback values
(EAR 0.25, MAR 0.50) are stored and the app warns that the driver should
recalibrate.

---

## Tech stack

| Area | Library |
|---|---|
| Video capture and drawing | OpenCV |
| Face landmarks | MediaPipe 0.10.14 (Tasks API, `face_landmarker.task`) |
| Phone detection | Ultralytics YOLOv8n 8.4.130 |
| Emotion recognition | DeepFace on TensorFlow 2.16.1 / tf-keras 2.16.0 |
| Maths | NumPy 1.26.4, SciPy |
| Audio alerts | pygame (WAV beeps), pyttsx3 (offline text-to-speech) |
| SOS messaging | Twilio, python-dotenv |
| Storage | SQLite (Python standard library) |

---

## Project structure

```
.
├── app.py                      # Entry point: menu, calibration, main driving loop, HUD
├── requirements.txt
├── .env.template               # Twilio credential placeholders (copy to .env)
├── assets/                     # Alert sounds: soft.wav, medium.wav, critical.wav
├── models/                     # face_landmarker.task, yolov8n.pt (auto-downloaded)
├── notes/
│   └── decision.md             # Landmark index decisions shared by the team
├── data/                       # Created at runtime (profiles + session database)
└── src/
    ├── vision/
    │   ├── webcam.py           # Shared webcam helper for module tests
    │   ├── face_mesh.py        # MediaPipe Face Landmarker wrapper -> landmark dict
    │   ├── eye_ear.py          # Eye Aspect Ratio
    │   ├── mouth_mar.py        # Mouth Aspect Ratio
    │   ├── head_pose.py        # Yaw / pitch / roll via solvePnP
    │   ├── phone_yolo.py       # YOLOv8n "cell phone" detections
    │   └── calibration.py      # Per-driver baseline recording and loading
    ├── brain/
    │   ├── phone_detector.py   # Turns raw detections into a stable phone flag
    │   ├── emotion_detection.py# DeepFace dominant emotion
    │   └── fusion.py           # Weighted score -> SAFE / WARNING / DROWSY / CRITICAL
    ├── alerts/
    │   ├── beep.py             # Plays the three alert sounds
    │   ├── voice.py            # Spoken warnings
    │   └── sos.py              # Emergency SMS via Twilio
    └── data/
        └── db.py               # SQLite sessions + events
```

---

## Setup

Requirements: **Python 3.9 – 3.12** (supported range of TensorFlow 2.16 and
MediaPipe 0.10.14; developed on 3.11), a webcam, and speakers. Developed and
tested on Windows.

```bash
git clone https://github.com/GHOSZTYY/DRIVER-DROWSINESS-DISTRACTION-DETECTION-SYSTEM.git
cd DRIVER-DROWSINESS-DISTRACTION-DETECTION-SYSTEM

python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

### Optional: SOS SMS

Copy the template and fill in your Twilio details:

```bash
copy .env.template .env           # Windows
# cp .env.template .env           # macOS / Linux
```

```
TWILIO_ACCOUNT_SID=your_sid_here
TWILIO_AUTH_TOKEN=your_token_here
TWILIO_FROM=your_twilio_number
TWILIO_TO=your_personal_number
```

Without a `.env` file the app still runs; it prints a warning and skips the
SMS. Never commit your `.env` file.

### Models

- `models/face_landmarker.task` is downloaded automatically on first run if it
  is missing.
- `models/yolov8n.pt` is downloaded automatically by Ultralytics on first run.
- DeepFace downloads its emotion weights on first use.

The first run therefore needs an internet connection.

---

## Running the app

Run from the **repository root**, because the sound files, profiles and
database use paths relative to it:

```bash
python app.py
```

1. Enter the driver's name (press Enter for `Driver`).
2. Press **`c`** to calibrate — look straight ahead and act naturally for
   5 seconds.
3. Press **`s`** to start the driving session.
4. Press **`q`** to stop driving and return to the menu; press **`q`** again
   to quit.

| Key | Where | Action |
|---|---|---|
| `c` | Menu | Calibrate (5 s baseline) |
| `s` | Menu | Start driving (needs a profile) |
| `q` | Driving | Stop and return to menu |
| `q` | Menu | Quit |

Start-up takes around 15 seconds while TensorFlow, DeepFace and YOLO load.

---

## Testing individual modules

Most modules have a small live test you can run on its own:

```bash
python src/vision/webcam.py            # raw webcam feed
python src/vision/face_mesh.py         # landmarks drawn on your face
python src/vision/eye_ear.py           # live EAR + blink count
python src/vision/mouth_mar.py         # live MAR + yawn flag
python src/vision/head_pose.py         # live yaw / pitch / roll
python src/vision/phone_yolo.py        # phone detection
python src/alerts/beep.py              # plays the medium alert
python src/alerts/voice.py             # speaks a test warning
python src/alerts/sos.py               # sends a test SMS (needs .env)
python src/data/db.py                  # creates the DB and writes sample rows
```

---

## Session logging (SQLite)

Database file: `data/sessions/driver_safety.db`

**`sessions`** — one row per drive

| Column | Meaning |
|---|---|
| `id` | Session ID |
| `driver_name` | Driver |
| `started_at`, `ended_at` | ISO timestamps |
| `total_alerts` | Alerts raised in the session |

**`events`** — one row per alert

| Column | Meaning |
|---|---|
| `session_id` | Links to `sessions.id` |
| `timestamp` | When it happened |
| `alert_level` | `low`, `medium` or `critical` |
| `ear_score`, `mar_score`, `yaw_angle` | Readings at that moment |
| `phone_detected` | 1 if a phone was seen |
| `emotion_label` | Detected emotion |

`get_all_sessions()` and `get_events_for_session()` in `src/data/db.py` read
the history back, ready for a dashboard.

---

## Configuration and tuning

| Setting | File | Default | Meaning |
|---|---|---|---|
| `EYES_CLOSED_DROWSY_SEC` / `EYES_CLOSED_CRITICAL_SEC` | `src/brain/fusion.py` | 1.5 s / 3 s | Eyes-closed time before DROWSY / CRITICAL |
| `NO_FACE_DROWSY_SEC` / `NO_FACE_CRITICAL_SEC` | `src/brain/fusion.py` | 2 s / 5 s | Face-missing time before DROWSY / CRITICAL |
| `BEEP_REPEAT_SEC` | `app.py` | DROWSY 2 s, CRITICAL 1 s | How often the beep repeats while the alert lasts |
| `VOICE_REPEAT_SEC` | `app.py` | 6 s | How often the spoken warning repeats while CRITICAL |
| `MIN_BEEP_GAP_SEC` | `app.py` | 1 s | Shortest time between two beeps |
| `EPISODE_END_SEC` | `app.py` | 3 s | Calm time before a new alert is logged again |
| `EMOTION_EVERY_N_FRAMES` | `app.py` | 15 | How often DeepFace runs |
| EAR ratio | `app.py` | 0.8 × baseline | Below this counts as closing |
| MAR ratio | `app.py` | 1.5 × baseline | Above this counts as a yawn |
| `YAW_THRESHOLD` / `PITCH_THRESHOLD` | `src/vision/head_pose.py` | 25° / 20° | Head turn limits |
| `DISTRACT_FRAMES` | `src/vision/head_pose.py` | 45 | Frames of looking away before flagging |
| `CAMERA_PITCH_OFFSET` | `src/vision/head_pose.py` | 25° | Compensates for a camera mounted above eye level — set to 0 for a camera at eye level |
| Phone thresholds | `app.py` (`PhoneDetector`) | conf 0.5, 10 frames | Phone confidence and persistence |
| Score bands | `src/brain/fusion.py` | 0 / 1–4 / 5–7 / 8+ | Status boundaries |

---

## Known limitations

- The phone and head-pose checks still count frames and assume roughly
  30 FPS; on slower machines the real time before those alerts is longer.
  (Eye and face checks use real seconds.)
- A face hidden for 2+ seconds for an innocent reason (drinking from a bottle,
  a long look over the shoulder) is still treated as `DROWSY`, and 5+ seconds
  as `CRITICAL`, because the system cannot tell this apart from a head drop.
- Squinting (bright sun, laughing hard) can look like closed eyes; held for
  1.5+ seconds it raises a false `DROWSY`.
- Head-pose thresholds assume the camera is roughly in front of the driver;
  `CAMERA_PITCH_OFFSET` must match the actual mounting angle.
- Emotion recognition from a single webcam frame is noisy and is only used as
  a small extra weight.
- This is an academic prototype. It is not a certified safety device and must
  not replace an alert, rested driver.

---

## Team

| Member | GitHub |
|---|---|
| Mridul Binjrajka | [@GHOSZTYY](https://github.com/GHOSZTYY) |
| Sarthak Khandelwal | [@sarthak-k08](https://github.com/sarthak-k08) |
| Samridhi Kumari | — |
