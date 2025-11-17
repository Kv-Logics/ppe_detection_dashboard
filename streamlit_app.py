# ============================================================
# STREAMLIT DASHBOARD WITH LIVE FEED & VIDEO UPLOAD
# ============================================================
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import tempfile
import time
import os

# Import the alert function from the external module
try:
    from alert_manager import send_email_alert
except ImportError:
    st.error("Error: Could not import 'send_email_alert'. Make sure 'alert_manager.py' is in the same directory.")
    send_email_alert = lambda *args: False 

# Page config
st.set_page_config(page_title="PPE Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (REVISED: Removes all custom colors for maximum theme compatibility)
st.markdown("""
    <style>
    /* 1. Header and Subheader Styling: Removed color to rely on Streamlit's default theme */
    h1, h2, h3 { 
        /* color: #000000; - REMOVED */
    }
    
    /* 2. Metric/Card Styling: Rely on Streamlit's color defaults for contrast */
    .stMetric > div[data-testid="stMetricValue"] {
        font-size: 1.8rem; 
        font-weight: 700;
        /* color: #000000; - REMOVED */
    }
    .stMetric > div[data-testid="stMetricLabel"] {
        /* color: #555555; - REMOVED */
    }
    
    /* 3. Sidebar Header Styling: Rely on Streamlit's color defaults */
    .css-1d391kg { 
        /* color: #000000; - REMOVED */
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* 4. Plot Text: Ensure plots use transparent background for theme compatibility */
    .plot-container, .modebar {
        font-family: inherit !important;
        color: inherit !important;
    }
    
    /* 5. General App Background: Ensure no custom background interferes */
    .main {
        background-color: transparent; 
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state (omitted for brevity)
if 'processed_results' not in st.session_state: st.session_state.processed_results = []
if 'violation_logs' not in st.session_state: st.session_state.violation_logs = []
if 'detection_stats' not in st.session_state: st.session_state.detection_stats = {}
if 'run_webcam' not in st.session_state: st.session_state.run_webcam = False
if 'all_detections' not in st.session_state: st.session_state.all_detections = []
if 'all_frames' not in st.session_state: st.session_state.all_frames = 0 

# Load YOLOv8 model (omitted for brevity)
@st.cache_resource
def load_model(model_path='best.pt'):
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.info("Please place your 'best.pt' file in the same directory as this script")
            return None
        model = YOLO(model_path)
        st.success(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# PPE Detection and Rule Engine 
def detect_ppe_violations(frame, model, frame_num, conf_threshold, recipient=None, sender=None, password=None):
    """
    Run YOLOv8 detection, apply rule engine, and trigger email alert for high-severity violations.
    """
    results = model(frame, conf=conf_threshold) 
    violations = []
    detections = []
    
    # Add Timestamp Overlay
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Use white text for timestamp (readable on most video backgrounds)
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            
            if conf >= conf_threshold:
                detections.append({'class': class_name, 'confidence': conf, 'bbox': box.xyxy[0].tolist()})
                
                # Rule Engine: Check for violations
                if 'NO-' in class_name or 'NoVest' in class_name or 'NoHardhat' in class_name or 'NoMask' in class_name:
                    severity = 'High' if class_name in ['NO-Hardhat', 'NO-Mask', 'NoHardhat', 'NoMask'] else 'Medium'
                    violation = {
                        'timestamp': timestamp, 'frame_num': frame_num, 'violation_type': class_name,
                        'confidence': round(conf, 4), 'severity': severity, 'status': 'Unresolved'
                    }
                    violations.append(violation)
                    
                    # ALERT TRIGGER
                    if violation['severity'] == 'High':
                        send_email_alert(recipient, sender, password, violation)

    return detections, violations, results[0].plot()

# Main dashboard
st.markdown("<h1 style='text-align: center;'>PPE Detection Monitoring Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")
    
    # Model selection
    st.subheader("Model Settings")
    model_path = st.text_input("Model Path", value="best.pt", help="Path to your YOLOv8 model file")
    
    st.markdown("---")
    
    # Mode Selection (Default: Process Video, which is index 1)
    mode = st.radio("Select Application Mode", 
        ["View Statistics", "Process Video (Upload)", "Live Feed (Webcam/URL)"], 
        index=1 # Set "Process Video (Upload)" as default
    )
    
    st.markdown("---")
    
    # --- Email Alert Configuration (Frontend Input) ---
    st.header("Email Alert Configuration")
    recipient_email = st.text_input("Recipient Email", value="", help="Email address to receive HIGH-severity alerts.")
    sender_email = st.text_input("Sender Gmail Address", value="", help="Your Gmail address for sending alerts.")
    app_password = st.text_input("Google App Password", type="password", value="", help="Required for sending alerts via Gmail SMTP.")
    
    if recipient_email and not app_password:
        st.warning("App Password Required for sending alerts via Gmail.")
    # --- End of Email Alert Configuration ---

    st.markdown("---")
    st.header("System Information")
    st.info(f"Status: Online\n\nLast Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load model
model = load_model(model_path)

if model is None:
    st.error("Cannot proceed without a valid model. Please check your model path.")
    st.stop()

# --- Horizontal Rule to separate sidebar from main content ---
st.markdown("---")

# ============================================================
# MODE 1: PROCESS VIDEO (Upload) - Default Mode
# ============================================================
if mode == "Process Video (Upload)":
    st.header("Video Processing (Upload)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file:
            # File handling logic
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name
            
            st.success("Video uploaded successfully!")
            st.video(video_path)
            
            st.subheader("Processing Settings")
            process_every_n = st.slider("Process every N frames", 1, 10, 5, help="Higher = faster but less accurate")
            conf_threshold = st.slider("Confidence Threshold (Detection & Logging)", 0.3, 0.9, 0.5, 0.05)
            save_output = st.checkbox("Save processed video", value=True)
            
            if st.button("Start Processing", type="primary", use_container_width=True):
                progress_bar = st.progress(0); status_text = st.empty(); video_placeholder = st.empty()
                st.session_state.processed_results = []; st.session_state.violation_logs = []; st.session_state.all_detections = []; st.session_state.all_frames = 0 
                detection_counts = {}
                
                if recipient_email and sender_email and app_password:
                    st.info(f"Email alerts ENABLED. Sending High Severity alerts to: {recipient_email}")
                else:
                    st.info("Email alerts DISABLED for video processing.")
                
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                output_path = None; out = None
                if save_output:
                    output_path = 'output_processed.mp4'; fourcc = cv2.VideoWriter_fourcc(*'mp4v'); out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0; processed_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    frame_count += 1
                    
                    if frame_count % process_every_n == 0:
                        processed_count += 1
                        
                        detections, violations, annotated_frame = detect_ppe_violations(
                            frame, model, frame_count, conf_threshold,
                            recipient_email, sender_email, app_password
                        )
                        
                        st.session_state.all_detections.extend(detections)
                        for det in detections: detection_counts[det['class']] = detection_counts.get(det['class'], 0) + 1
                        st.session_state.processed_results.append({'frame': frame_count, 'detections': len(detections), 'violations': len(violations)})
                        st.session_state.violation_logs.extend(violations)
                        
                        if processed_count % 10 == 0: video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                        if save_output and out is not None: out.write(annotated_frame)
                    else:
                        if save_output and out is not None: out.write(frame)
                    
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: Frame {frame_count}/{total_frames} | Violations: {len(st.session_state.violation_logs)}")
                
                cap.release();
                if out is not None: out.release()
                
                st.session_state.detection_stats = detection_counts; st.session_state.all_frames = processed_count 
                
                progress_bar.progress(1.0); status_text.text(f"Complete! Processed {frame_count} frames.")
                st.success(f"Processing complete! Total violations: {len(st.session_state.violation_logs)}")
                
                if save_output and os.path.exists(output_path):
                    st.subheader("Processed Video"); st.video(output_path)
                    with open(output_path, 'rb') as f:
                        st.download_button(label="Download Processed Video", data=f, file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4", mime="video/mp4", use_container_width=True)
    
    with col2:
        if st.session_state.processed_results:
            st.subheader("Processing Results")
            total_detections = sum([r['detections'] for r in st.session_state.processed_results]); total_violations = sum([r['violations'] for r in st.session_state.processed_results]); total_frames = len(st.session_state.processed_results)
            avg_confidence = 0
            if st.session_state.all_detections: avg_confidence = sum([d['confidence'] for d in st.session_state.all_detections]) / len(st.session_state.all_detections)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Frames Processed", total_frames); col_b.metric("Total Detections", total_detections); col_c.metric("Total Violations", total_violations)
            st.metric("Average Confidence", f"{avg_confidence*100:.2f}%", help="Average confidence of all detections")
            if st.session_state.detection_stats:
                # Plots use transparent background for theme compatibility
                fig = px.bar(x=list(st.session_state.detection_stats.keys()), y=list(st.session_state.detection_stats.values()), title="Detections by Class", labels={'x': 'Class', 'y': 'Count'}, color=list(st.session_state.detection_stats.values()), color_continuous_scale='viridis')
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODE 2: LIVE FEED (Webcam/URL)
# ============================================================
elif mode == "Live Feed (Webcam/URL)":
    st.header("Live Feed Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Stream Controls")
        
        # Source Selection: Combines Webcam and URL input
        source_type = st.radio("Select Video Source", ["Webcam (Index 0)", "Webcam (Index 1)", "External URL (RTSP/HTTP)"], help="Select a local webcam index or provide a network stream URL.")
        
        live_source = 0
        if source_type == "External URL (RTSP/HTTP)":
            live_source = st.text_input("Enter Stream URL", value="rtsp://...", help="Input RTSP or HTTP video stream address.")
            if not live_source or live_source == "rtsp://...": live_source = None
        elif source_type == "Webcam (Index 1)": 
            live_source = 1
        # If "Webcam (Index 0)" is selected, live_source remains 0.

        live_conf_threshold = st.slider("Detection Confidence Threshold", min_value=0.3, max_value=0.9, value=0.5, step=0.05, help="Minimum confidence for drawing boxes and logging violations.")
        frame_skip = st.slider("Processing Skip Factor (Optimization)", min_value=1, max_value=5, value=3, step=1, help="Process every Nth frame.")
        
        st.markdown("---"); stats_placeholder = st.empty(); violations_placeholder = st.empty()
        
    with col1:
        if st.button("Start Stream" if not st.session_state.run_webcam else "Stop Stream", type="primary", use_container_width=True):
            st.session_state.run_webcam = not st.session_state.run_webcam
        FRAME_WINDOW = st.empty()
    
    if st.session_state.run_webcam and live_source is not None:
        
        if recipient_email and sender_email and app_password:
            st.warning("Email alerts ENABLED for this live stream. Only HIGH severity will trigger emails.")
        
        camera = cv2.VideoCapture(live_source) 
        if not camera.isOpened():
            st.error(f"Could not access stream/webcam at source: **{live_source}**."); st.session_state.run_webcam = False
        else:
            detection_history = []; violation_history = []; frame_count = 0
            while st.session_state.run_webcam:
                ret, frame = camera.read();
                if not ret: st.error("Stream finished or failed to read frame."); break
                frame_count += 1
                
                if frame_count % frame_skip == 0: 
                    detections, violations, annotated_frame = detect_ppe_violations(
                        frame, model, frame_count, live_conf_threshold,
                        recipient_email, sender_email, app_password
                    )
                    detection_history.extend(detections); violation_history.extend(violations)
                    FRAME_WINDOW.image(annotated_frame, channels="BGR", use_column_width=True) 
                    
                    with stats_placeholder.container():
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Frames", frame_count); col_b.metric("Detections", len(detection_history)); col_c.metric("Violations", len(violation_history))
                    
                    if violations:
                        with violations_placeholder.container():
                            st.warning(f"**{len(violations)} violation(s) detected!**")
                            for v in violations[-3:]: st.write(f"- {v['violation_type']} (Conf: {v['confidence']*100:.1f}%)")
                
                time.sleep(0.01)
            
            camera.release(); st.session_state.run_webcam = False; st.success("Stream stopped")
    elif live_source is None and st.session_state.run_webcam: st.error("Cannot start stream without a valid URL.")
    elif not st.session_state.run_webcam: st.info("Click 'Start Stream' to begin live detection")

# ============================================================
# MODE 3: VIEW STATISTICS
# ============================================================
elif mode == "View Statistics":
    st.header("Detection Statistics")
    
    if st.session_state.processed_results:
        total_frames = st.session_state.all_frames; total_detections = sum([r['detections'] for r in st.session_state.processed_results]); total_violations = len(st.session_state.violation_logs)
        avg_confidence = 0; compliance_percent = 0
        if total_detections > 0: avg_confidence = sum([d['confidence'] for d in st.session_state.all_detections]) / len(st.session_state.all_detections); violation_rate = total_violations / total_detections; compliance_percent = (1 - violation_rate) * 100
        
        st.subheader("KPI Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", total_frames); col2.metric("Total Detections", total_detections); col3.metric("Violations Detected", total_violations, delta=f"{total_violations} total"); col4.metric("Compliance Rate", f"{compliance_percent:.2f}%", delta_color="inverse")
        st.metric("Average Confidence", f"{avg_confidence*100:.2f}%")

        st.subheader("Compliance Overview")
        if total_detections > 0:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = compliance_percent, domain = {'x': [0, 1], 'y': [0, 1]}, title = {'text': "PPE Compliance %"}, gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#667eea"}, 'steps': [{'range': [0, 70], 'color': "red"}, {'range': [70, 90], 'color': "yellow"}, {'range': [90, 100], 'color': "green"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': compliance_percent}}))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        st.subheader("Violations Trend Over Frame Number")
        if st.session_state.violation_logs:
            # Line chart
            df_logs = pd.DataFrame(st.session_state.violation_logs); violations_per_frame = df_logs.groupby('frame_num').size().reset_index(name='Count')
            fig_time = px.line(violations_per_frame, x='frame_num', y='Count', title='Violations Over Frame Number', labels={'frame_num': 'Frame Number', 'Count': 'Number of Violations'}, line_shape='spline', color_discrete_sequence=['#ff4b4b'])
            fig_time.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_time, use_container_width=True)
            
        st.subheader("Detection Distribution")
        if st.session_state.detection_stats:
            # Pie chart
            fig = px.pie(values=list(st.session_state.detection_stats.values()), names=list(st.session_state.detection_stats.keys()), title="Detections by Class Type", hole=0.4)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No detection results found. Please process a video first in the 'Process Video (Upload)' mode.")
        st.info("Sample data shown below:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", "82"); col2.metric("Total Detections", "639"); col3.metric("Violations Detected", "256"); col4.metric("Avg Confidence", "78.00%")

# ============================================================
# VIOLATION LOGS
# ============================================================
if st.session_state.violation_logs:
    st.markdown("---"); st.header("Violation Logs")
    df = pd.DataFrame(st.session_state.violation_logs); st.dataframe(df, use_container_width=True, height=400)
    st.subheader("Violations Summary"); violation_counts = df['violation_type'].value_counts()
    fig = px.bar(x=violation_counts.index, y=violation_counts.values, title="Violations by Type", labels={'x': 'Violation Type', 'y': 'Count'}, color=violation_counts.values, color_continuous_scale='reds')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)
    csv = df.to_csv(index=False)
    st.download_button(label="Export Violations to CSV", data=csv, file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>SafetyEye - AI-Powered PPE Detection System | Built with YOLOv8 & Streamlit</p>", unsafe_allow_html=True)