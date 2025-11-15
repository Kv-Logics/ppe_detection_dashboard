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

# Page config
st.set_page_config(page_title="PPE Detection Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    h1, h2, h3 { color: #ffffff; }
    .metric-card { 
        background: rgba(255,255,255,0.1); 
        padding: 20px; 
        border-radius: 10px; 
        backdrop-filter: blur(10px);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'violation_logs' not in st.session_state:
    st.session_state.violation_logs = []
if 'detection_stats' not in st.session_state:
    st.session_state.detection_stats = {}
if 'run_webcam' not in st.session_state:
    st.session_state.run_webcam = False
if 'all_detections' not in st.session_state:
    st.session_state.all_detections = []

# Load YOLOv8 model
@st.cache_resource
def load_model(model_path='best.pt'):
    """Load YOLOv8 model from local path"""
    try:
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found: {model_path}")
            st.info("Please place your 'best.pt' file in the same directory as this script")
            return None
        
        model = YOLO(model_path)
        st.success(f"✅ Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

# PPE Detection and Rule Engine
def detect_ppe_violations(frame, model, frame_num):
    """Run YOLOv8 detection and apply rule engine for violations"""
    results = model(frame, conf=0.5)
    
    violations = []
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            detections.append({
                'class': class_name,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2]
            })
            
            # Rule Engine: Check for violations
            if 'NO-' in class_name or 'NoVest' in class_name or 'NoHardhat' in class_name or 'NoMask' in class_name:
                severity = 'High' if class_name in ['NO-Hardhat', 'NO-Mask', 'NoHardhat', 'NoMask'] else 'Medium'
                violation = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'frame_num': frame_num,
                    'violation_type': class_name,
                    'confidence': round(conf, 4),
                    'severity': severity,
                    'status': 'Unresolved'
                }
                violations.append(violation)
    
    return detections, violations, results[0].plot()

# Main dashboard
st.markdown("<h1 style='text-align: center;'>PPE Detection Monitoring Dashboard</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    # Model selection
    st.subheader(" Model Settings")
    model_path = st.text_input("Model Path", value="best.pt", help="Path to your YOLOv8 model file")
    
    if not os.path.exists(model_path):
        st.warning(f"⚠️ Model file '{model_path}' not found in current directory")
        st.info(f"Current directory: {os.getcwd()}")
    
    st.markdown("---")
    
    mode = st.radio("Select Mode", ["📊 View Statistics", "📹 Process Video", "📷 Webcam (Live Feed)"])
    
    st.markdown("---")
    st.header("📊 System Info")
    st.info(f"**Status:** Online\n\n**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Load model
model = load_model(model_path)

if model is None:
    st.error("⚠️ Cannot proceed without a valid model. Please check your model path.")
    st.stop()

# ============================================================
# MODE 1: PROCESS VIDEO
# ============================================================
if mode == "📹 Process Video":
    st.header("📹 Video Processing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mov', 'mkv'])
        
        if uploaded_file:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name
            
            st.success("✅ Video uploaded successfully!")
            st.video(video_path)
            
            # Processing settings
            st.subheader("⚙️ Processing Settings")
            process_every_n = st.slider("Process every N frames", 1, 10, 5, help="Higher = faster but less accurate")
            conf_threshold = st.slider("Confidence Threshold", 0.3, 0.9, 0.5, 0.05)
            save_output = st.checkbox("Save processed video", value=True)
            
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                video_placeholder = st.empty()
                
                # Reset session state
                st.session_state.processed_results = []
                st.session_state.violation_logs = []
                st.session_state.all_detections = []
                detection_counts = {}
                
                # Process video
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Video writer for output
                output_path = None
                out = None
                if save_output:
                    output_path = 'output_processed.mp4'
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                frame_count = 0
                processed_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Process selected frames
                    if frame_count % process_every_n == 0:
                        processed_count += 1
                        
                        # Run detection
                        detections, violations, annotated_frame = detect_ppe_violations(frame, model, frame_count)
                        
                        # Store all detections for confidence calculation
                        st.session_state.all_detections.extend(detections)
                        
                        # Update stats
                        for det in detections:
                            detection_counts[det['class']] = detection_counts.get(det['class'], 0) + 1
                        
                        st.session_state.processed_results.append({
                            'frame': frame_count,
                            'detections': len(detections),
                            'violations': len(violations)
                        })
                        
                        st.session_state.violation_logs.extend(violations)
                        
                        # Display frame (every 10th processed frame to avoid slowdown)
                        if processed_count % 10 == 0:
                            video_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                        
                        # Write to output video
                        if save_output and out is not None:
                            out.write(annotated_frame)
                    else:
                        # Write original frame to output
                        if save_output and out is not None:
                            out.write(frame)
                    
                    # Update progress
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: Frame {frame_count}/{total_frames} | Violations: {len(st.session_state.violation_logs)}")
                
                cap.release()
                if out is not None:
                    out.release()
                
                st.session_state.detection_stats = detection_counts
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Complete! Processed {frame_count} frames.")
                st.success(f"🎉 Processing complete! Total violations: {len(st.session_state.violation_logs)}")
                st.balloons()
                
                # Show output video
                if save_output and os.path.exists(output_path):
                    st.subheader("📹 Processed Video")
                    st.video(output_path)
                    
                    # Download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="⬇️ Download Processed Video",
                            data=f,
                            file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
    
    with col2:
        if st.session_state.processed_results:
            st.subheader("📊 Processing Results")
            
            # Summary metrics
            total_detections = sum([r['detections'] for r in st.session_state.processed_results])
            total_violations = sum([r['violations'] for r in st.session_state.processed_results])
            total_frames = len(st.session_state.processed_results)
            
            # Calculate average confidence from ALL detections
            avg_confidence = 0
            if st.session_state.all_detections:
                avg_confidence = sum([d['confidence'] for d in st.session_state.all_detections]) / len(st.session_state.all_detections)
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Frames Processed", total_frames)
            col_b.metric("Total Detections", total_detections)
            col_c.metric("Total Violations", total_violations)
            
            # Show average confidence
            st.metric("🎯 Average Confidence", f"{avg_confidence*100:.2f}%", 
                     help="Average confidence of all detections")
            
            # Detection chart
            if st.session_state.detection_stats:
                fig = px.bar(
                    x=list(st.session_state.detection_stats.keys()),
                    y=list(st.session_state.detection_stats.values()),
                    title="Detections by Class",
                    labels={'x': 'Class', 'y': 'Count'},
                    color=list(st.session_state.detection_stats.values()),
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig, use_container_width=True)

# ============================================================
# MODE 2: WEBCAM LIVE FEED
# ============================================================
elif mode == "📷 Webcam (Live Feed)":
    st.header("📷 Live Webcam Feed")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Start/Stop button
        if st.button("🔴 Start Webcam" if not st.session_state.run_webcam else "⏹️ Stop Webcam", 
                     type="primary", use_container_width=True):
            st.session_state.run_webcam = not st.session_state.run_webcam
        
        FRAME_WINDOW = st.empty()
    
    with col2:
        stats_placeholder = st.empty()
        violations_placeholder = st.empty()
    
    if st.session_state.run_webcam:
        camera = cv2.VideoCapture(0)
        
        if not camera.isOpened():
            st.error("❌ Could not access webcam. Please check your camera connection.")
            st.session_state.run_webcam = False
        else:
            detection_history = []
            violation_history = []
            frame_count = 0
            
            while st.session_state.run_webcam:
                ret, frame = camera.read()
                if not ret:
                    st.error("Failed to read frame from webcam")
                    break
                
                frame_count += 1
                
                # Process every 3rd frame for performance
                if frame_count % 3 == 0:
                    # Run detection
                    detections, violations, annotated_frame = detect_ppe_violations(
                        frame, model, frame_count
                    )
                    
                    detection_history.extend(detections)
                    violation_history.extend(violations)
                    
                    # Display annotated frame
                    FRAME_WINDOW.image(annotated_frame, channels="BGR", use_column_width=True)
                    
                    # Display live stats
                    with stats_placeholder.container():
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Frames", frame_count)
                        col_b.metric("Detections", len(detection_history))
                        col_c.metric("Violations", len(violation_history))
                    
                    # Show recent violations
                    if violations:
                        with violations_placeholder.container():
                            st.warning(f"⚠️ **{len(violations)} violation(s) detected!**")
                            for v in violations[-3:]:  # Show last 3
                                st.write(f"- {v['violation_type']} (Confidence: {v['confidence']*100:.1f}%)")
                
                # Add small delay to prevent UI freezing
                time.sleep(0.01)
            
            camera.release()
            st.success("✅ Webcam stopped")
    else:
        st.info("Click 'Start Webcam' to begin live detection")

# ============================================================
# MODE 3: VIEW STATISTICS
# ============================================================
elif mode == "📊 View Statistics":
    st.header("📊 Detection Statistics")
    
    # Check if we have processed data
    if st.session_state.processed_results:
        # Metrics
        total_frames = len(st.session_state.processed_results)
        total_detections = sum([r['detections'] for r in st.session_state.processed_results])
        total_violations = len(st.session_state.violation_logs)
        
        # Calculate average confidence from ALL detections (not just violations)
        avg_confidence = 0
        if st.session_state.all_detections:
            avg_confidence = sum([d['confidence'] for d in st.session_state.all_detections]) / len(st.session_state.all_detections)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", total_frames)
        col2.metric("Total Detections", total_detections)
        col3.metric("Violations Detected", total_violations)
        col4.metric("Avg Confidence", f"{avg_confidence*100:.2f}%")
        
        # Charts
        st.subheader("Detection Distribution")
        
        if st.session_state.detection_stats:
            fig = px.pie(
                values=list(st.session_state.detection_stats.values()),
                names=list(st.session_state.detection_stats.keys()),
                title="Detections by Class Type",
                hole=0.4
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("📊 No detection results found. Please process a video first in the '📹 Process Video' mode.")
        
        # Show sample statistics
        st.info("Sample data shown below:")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Frames", "82")
        col2.metric("Total Detections", "639")
        col3.metric("Violations Detected", "256")
        col4.metric("Avg Confidence", "78.00%")

# ============================================================
# VIOLATION LOGS (Shown in all modes)
# ============================================================
if st.session_state.violation_logs:
    st.markdown("---")
    st.header("⚠️ Violation Logs")
    
    df = pd.DataFrame(st.session_state.violation_logs)
    
    # Display dataframe
    st.dataframe(df, use_container_width=True, height=400)
    
    # Summary by type
    st.subheader("📊 Violations Summary")
    violation_counts = df['violation_type'].value_counts()
    
    fig = px.bar(
        x=violation_counts.index,
        y=violation_counts.values,
        title="Violations by Type",
        labels={'x': 'Violation Type', 'y': 'Count'},
        color=violation_counts.values,
        color_continuous_scale='reds'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Export
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Export Violations to CSV",
        data=csv,
        file_name=f"violations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888;'>SafetyEye - AI-Powered PPE Detection System | Built with YOLOv8 & Streamlit</p>", unsafe_allow_html=True)