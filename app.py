import streamlit as st
import cv2
import time
from src.vision.calibration import run_calibration, load_profile
# Importing Mridul's webcam function
try:
    from src.vision.webcam import get_frame
except ImportError:
    # Fallback just in case Mridul hasn't pushed this yet!
    get_frame = lambda: None 

# Week 1: Basic App Setup
st.set_page_config(page_title='Driver Safety')
st.title('🚗 Driver Safety Co-Pilot')

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
if st.sidebar.button('Calibrate (60s)'):
    st.sidebar.info("Calibrating... Look straight ahead and act naturally.")
    
    # Temporarily using placeholder lambda functions (0.5) until 
    # Mridul/Samriddhi finish pushing the actual Face Mesh math!
    get_placeholder_val = lambda: 0.5 
    
    new_profile = run_calibration(
        driver_name=name,
        get_ear_fn=get_placeholder_val,
        get_mar_fn=get_placeholder_val,
        get_pose_fn=get_placeholder_val,
        duration_sec=5  # Let's test with 5 seconds first before doing the full 60!
    )
    
    st.sidebar.success("Calibration Complete!")
    time.sleep(2)
    st.rerun() # This refreshes Streamlit so the new profile loads
st.sidebar.metric('FPS', '0') # We will calculate this properly later

# Week 2: Streamlit State Management
if 'running' not in st.session_state:
    st.session_state.running = False

# Week 2: Start/Stop Buttons
if st.button('Start'):
    st.session_state.running = True
if st.button('Stop'):
    st.session_state.running = False

frame_placeholder = st.empty()

# Week 2: Live Video Loop
if st.session_state.running:
    frame = get_frame()
    if frame is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, use_column_width=True)
    else:
        st.warning("Camera feed not available yet.")