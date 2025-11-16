import cv2
import streamlit as st
import tempfile
import time
from datetime import datetime

def run_detection_on_stream(source, detector, confidence_threshold, video_placeholder, stats_placeholders):
    """
    Processes a video stream (from file or webcam) and updates the Streamlit UI.
    
    Args:
        source: File path, URL, or webcam index (0).
        detector: An initialized PPEDetector object.
        confidence_threshold: The confidence threshold for detection.
        video_placeholder: st.empty() for displaying the video feed.
        stats_placeholders: A dictionary of st.empty() for displaying stats.
    """
    
    # Initialize state for this run
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False
    st.session_state.stop_processing = False
    
    # --- Video Source Setup ---
    is_webcam = (source == 0)
    cap = None
    
    try:
        if isinstance(source, str) and (source.startswith('rtsp://') or source.startswith('http://')):
            # RTSP/HTTP Stream
            cap = cv2.VideoCapture(source)
            st.info(f"Connecting to stream: {source}")
        elif is_webcam:
            # Webcam
            cap = cv2.VideoCapture(0)
            st.info("Accessing webcam...")
        else:
            # Uploaded Video File
            cap = cv2.VideoCapture(source)
            st.info("Processing uploaded video file...")
            
        if not cap or not cap.isOpened():
            st.error(f"Error: Could not open video source '{source}'.")
            return

        # --- Statistics Initialization ---
        total_frames = 0
        total_violations = 0
        total_persons = 0
        total_ppe_compliant = 0
        total_confidence = 0
        total_detections = 0
        start_time = time.time()

        if 'violation_logs' not in st.session_state:
            st.session_state.violation_logs = []

        # --- Main Processing Loop ---
        while cap.isOpened():
            # Check for stop signal
            if st.session_state.stop_processing:
                st.warning("Detection stopped by user.")
                break
                
            ret, frame = cap.read()
            if not ret:
                st.success("Video processing finished.")
                break

            total_frames += 1
            
            # Run detection on the frame
            annotated_frame, frame_stats, new_violations = detector.process_frame(
                frame, 
                confidence_threshold
            )
            
            # --- Update Stats ---
            total_violations += frame_stats['violations']
            total_persons += frame_stats['persons']
            total_ppe_compliant += frame_stats['ppe_compliant']
            total_confidence += frame_stats['confidence_sum']
            total_detections += frame_stats['detection_count']

            # Append new violations to the session state
            if new_violations:
                for v in new_violations:
                    v['frame_num'] = total_frames
                st.session_state.violation_logs.extend(new_violations)

            # --- Calculate Metrics (Module 2) ---
            elapsed_time = time.time() - start_time
            fps = total_frames / elapsed_time if elapsed_time > 0 else 0
            avg_confidence = (total_confidence / total_detections) * 100 if total_detections > 0 else 0

            # --- Update UI ---
            # 1. Update Video Feed (Module 1)
            video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            # 2. Update Stats Panel (Module 2)
            stats_placeholders['kpi_col1'].metric("Total Frames", f"{total_frames:,}")
            stats_placeholders['kpi_col2'].metric("Total Violations", f"{total_violations:,}")
            stats_placeholders['kpi_col3'].metric("Persons Detected", f"{total_persons:,}")
            stats_placeholders['kpi_col4'].metric("Est. FPS", f"{fps:.1f}")
            
    except Exception as e:
        st.error(f"An error occurred during video processing: {e}")
    finally:
        if cap:
            cap.release()
        st.session_state.stop_processing = False # Reset flag
        
        # --- Final Stats Update ---
        # This triggers the app to rerun, which will
        # render the final log table and pie chart.
        st.rerun()