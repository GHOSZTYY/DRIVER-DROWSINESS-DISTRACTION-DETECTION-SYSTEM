import streamlit as st
import cv2
import time
from src.alerts.sos import send_sos_alert
from src.vision.calibration import run_calibration, load_profile
# Importing Mridul's webcam function
try:
    from src.vision.webcam import get_frame
except ImportError:
    # Fallback just in case Mridul hasn't pushed this yet!
    get_frame = lambda: None 

# --- IMPORT TEAM AI MODELS ---
from src.vision.face_mesh import get_landmarks
from src.vision.eye_ear import compute_ear, LEFT_EYE, RIGHT_EYE
from src.vision.mouth_mar import compute_mar

# Week 1: Basic App Setup
st.set_page_config(page_title='Driver Safety')
st.title('🚗 Driver Safety Co-Pilot')
# Initialize the session state so Streamlit doesn't panic on first load
if 'running' not in st.session_state:
    st.session_state.running = False

# Week 2: Sidebar UI
name = st.sidebar.text_input('Your name', 'Driver')
st.sidebar.markdown("---")
st.sidebar.subheader("👤 Driver Profile")

# 1. Try to load an existing profile for this driver
profile = load_profile(name)
if profile:
    st.sidebar.success("✅ Profile Active")
    # Show the baseline stats to the team!
    st.sidebar.json(profile) 
else:
    st.sidebar.warning("⚠️ No profile found. Please calibrate.")

# 2. The Calibration Button
if st.sidebar.button('Calibrate (5s)'):
    st.sidebar.info("Calibrating... Look straight ahead and act naturally.")
    
    # Trigger the upgraded calibration file!
    new_profile = run_calibration(
        driver_name=name,
        duration_sec=5 
    )
    
    st.sidebar.success("Calibration Complete!")
    time.sleep(2)
    st.rerun() # Refreshes Streamlit so the new profile loads
st.sidebar.metric('FPS', '0') # We will calculate this properly later

# --- WEEK 6: MAIN INTEGRATION LOOP ---
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button('🟢 Start Driving'):
        if not profile:
            st.error("Crucial: You must calibrate your baseline before driving!")
        else:
            st.session_state.running = True
            st.rerun()

with col2:
    if st.button('🛑 Stop Driving'):
        st.session_state.running = False
        st.rerun()

if st.session_state.running:
    st.success("Co-Pilot Active. Monitoring driver...")
    
    video_placeholder = st.empty()
    alert_placeholder = st.empty()
    fps_placeholder = st.sidebar.empty()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    drowsy_frames = 0
    DROWSY_THRESHOLD_FRAMES = 15 
    
    # NEW: Yawn Tracking State
    # ==========================================
    # PRE-LOOP SETUP (MEMORY & TIMERS)
    # ==========================================
    total_yawns = 0
    is_yawning = False
    
    prev_time = time.time()
    frame_count = 0
    alert_frames = 0 # NEW: This keeps the red alert on screen smoothly!

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera disconnected.")
            break
            
        frame_count += 1
            
        # ==========================================
        # 1. RUN THE AI MODELS
        # ==========================================
        landmarks = get_landmarks(frame)
        
        if landmarks:
            left = compute_ear(landmarks, LEFT_EYE)
            right = compute_ear(landmarks, RIGHT_EYE)
            current_ear = (left + right) / 2.0
            current_mar = compute_mar(landmarks)
        else:
            current_ear = profile['ear_mean'] 
            current_mar = profile['mar_mean']
            
        # ==========================================
        # 2. THE BRAIN
        # ==========================================
        # Eyes
        if current_ear < (profile['ear_mean'] * 0.8):
            drowsy_frames += 1
        else:
            drowsy_frames = 0 
            
        # Mouth
        if current_mar > (profile['mar_mean'] * 1.5):
            if not is_yawning: 
                total_yawns += 1
                is_yawning = True
        else:
            is_yawning = False 
            
        # ==========================================
        # 3. FIRE THE ALERTS (STABILIZED)
        # ==========================================
        # If danger is detected, set a 30-frame timer so the alert stays on screen
        if drowsy_frames >= DROWSY_THRESHOLD_FRAMES or total_yawns > 3:
            alert_frames = 30 
            if total_yawns > 3:
                total_yawns = 0 # Reset yawns so it doesn't get stuck!
                
        # Show the red alert as long as the timer is running
        if alert_frames > 0:
            alert_placeholder.error("🚨 CRITICAL DROWSINESS DETECTED! WAKE UP! 🚨")
            alert_frames -= 1
        else:
            alert_placeholder.success("✅ Driver is alert. Eyes on the road.")
            
        # ==========================================
        # 4. VIDEO & FPS
        # ==========================================
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 0.001) # +0.001 prevents a math crash!
        prev_time = curr_time
        
        # Only send the FPS to the browser every 10 frames so it doesn't freeze
        if frame_count % 10 == 0:
            fps_placeholder.metric('FPS', f"{int(fps)}")

    cap.release()