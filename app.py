import streamlit as st
import cv2
import time

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