import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import json
import os

class PPEDetector:
    """
    PPE Detection Module using YOLOv8
    Detects construction site safety compliance including helmets, vests, and masks
    Based on Construction Site Safety Image Dataset from Roboflow
    """
    
    def __init__(self, model_path='best.pt', confidence_threshold=0.5):
        """
        Initialize the PPE detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence score for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Actual class names from your model
        self.class_names = {
            0: 'Hardhat',
            1: 'Mask',
            2: 'NO-Hardhat',
            3: 'NO-Mask',
            4: 'NO-Safety Vest',
            5: 'Person',
            6: 'Safety Cone',
            7: 'Safety Vest',
            8: 'machinery',
            9: 'vehicle'
        }
        
        # Violation classes (detected directly by model)
        self.violation_classes = {2, 3, 4}  # NO-Hardhat, NO-Mask, NO-Safety Vest
        
        # PPE classes
        self.ppe_classes = {0, 1, 7}  # Hardhat, Mask, Safety Vest
        
        # Initialize logs
        self.violation_log = []
        self.alert_log = []
        
    def detect_frame(self, frame):
        """
        Run detection on a single frame
        
        Args:
            frame: OpenCV image frame (BGR format)
            
        Returns:
            annotated_frame: Frame with bounding boxes
            detections: List of detection objects
        """
        results = self.model(frame, conf=self.confidence_threshold)
        
        annotated_frame = results[0].plot()
        detections = []
        
        for result in results[0].boxes.data:
            x1, y1, x2, y2, conf, cls = result
            detection = {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf),
                'class_id': int(cls),
                'class_name': self.class_names.get(int(cls), 'Unknown')
            }
            detections.append(detection)
        
        return annotated_frame, detections
    
    def check_violations(self, detections):
        """
        Check for PPE violations based on detection results
        Model directly detects violation classes (NO-Hardhat, NO-Mask, NO-Safety Vest)
        
        Args:
            detections: List of detection objects
            
        Returns:
            violations: List of violation objects
        """
        violations = []
        
        for detection in detections:
            # Check if detection is a violation class
            if detection['class_id'] in self.violation_classes:
                violation = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'violation_type': detection['class_name'],
                    'confidence': detection['confidence'],
                    'severity': self.get_severity(detection['class_name']),
                    'bbox': detection['bbox']
                }
                violations.append(violation)
        
        return violations
    
    def get_severity(self, violation_type):
        """
        Get violation severity based on violation type
        
        Args:
            violation_type: Name of violation
            
        Returns:
            severity: 'High', 'Medium', or 'Low'
        """
        severity_map = {
            'NO-Hardhat': 'High',
            'NO-Safety Vest': 'High',
            'NO-Mask': 'Medium'
        }
        return severity_map.get(violation_type, 'Low')
    
    def get_statistics(self, detections):
        """
        Get detection statistics
        
        Args:
            detections: List of detection objects
            
        Returns:
            stats: Dictionary of statistics
        """
        stats = {
            'total_detections': len(detections),
            'persons': sum(1 for d in detections if d['class_id'] == 5),
            'hardhats': sum(1 for d in detections if d['class_id'] == 0),
            'masks': sum(1 for d in detections if d['class_id'] == 1),
            'safety_vests': sum(1 for d in detections if d['class_id'] == 7),
            'no_hardhats': sum(1 for d in detections if d['class_id'] == 2),
            'no_masks': sum(1 for d in detections if d['class_id'] == 3),
            'no_safety_vests': sum(1 for d in detections if d['class_id'] == 4),
            'safety_cones': sum(1 for d in detections if d['class_id'] == 6),
            'machinery': sum(1 for d in detections if d['class_id'] == 8),
            'vehicles': sum(1 for d in detections if d['class_id'] == 9)
        }
        return stats
    
    def log_violation(self, violation, worker_id='Unknown', location='Unknown', frame_path=''):
        """
        Log a violation to CSV file
        
        Args:
            violation: Violation object
            worker_id: Worker identifier
            location: Location/zone identifier
            frame_path: Path to saved frame image
        """
        log_entry = {
            'timestamp': violation['timestamp'],
            'worker_id': worker_id,
            'violation_type': violation['violation_type'],
            'confidence': round(violation['confidence'] * 100, 2),  # Convert to percentage
            'severity': violation['severity'],
            'location': location,
            'frame_path': frame_path,
            'resolved': False
        }
        
        self.violation_log.append(log_entry)
        
        # Save to CSV
        self.save_violation_log()
    
    def save_violation_log(self, filename='violations_log.csv'):
        """Save violation log to CSV file"""
        if self.violation_log:
            df = pd.DataFrame(self.violation_log)
            
            # Append to existing file or create new
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
                # Remove duplicates based on timestamp and violation_type
                df = df.drop_duplicates(subset=['timestamp', 'violation_type'], keep='last')
            
            df.to_csv(filename, index=False)
            print(f"✓ Violation log saved to {filename}")
    
    def generate_alert(self, violation, worker_id='Unknown', location='Unknown'):
        """
        Generate an alert for high-severity violations
        
        Args:
            violation: Violation object
            worker_id: Worker identifier
            location: Location/zone identifier
        """
        if violation['severity'] in ['High', 'Medium']:
            alert = {
                'timestamp': violation['timestamp'],
                'alert_type': 'PPE_VIOLATION',
                'worker_id': worker_id,
                'violation': violation['violation_type'],
                'confidence': round(violation['confidence'] * 100, 2),
                'location': location,
                'severity': violation['severity'],
                'status': 'ACTIVE'
            }
            
            self.alert_log.append(alert)
            self.save_alert_log()
            
            # Print alert to console
            print(f"🚨 ALERT: {violation['violation_type']} detected at {location} (Confidence: {alert['confidence']:.1f}%)")
    
    def save_alert_log(self, filename='alertlog.csv'):
        """Save alert log to CSV file"""
        if self.alert_log:
            df = pd.DataFrame(self.alert_log)
            
            if os.path.exists(filename):
                existing_df = pd.read_csv(filename)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_csv(filename, index=False)
    
    def process_video(self, video_path, output_path='output_video.mp4', save_frames=True, skip_frames=1):
        """
        Process entire video file and detect PPE violations
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            save_frames: Whether to save violation frames
            skip_frames: Process every Nth frame (1 = all frames, 2 = every other frame)
            
        Returns:
            stats: Processing statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        violation_count = 0
        violation_frames = []
        
        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Skip frames: {skip_frames} (processing every {skip_frames} frame(s))")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames for faster processing
            if frame_count % skip_frames != 0:
                out.write(frame)
                continue
            
            processed_count += 1
            
            # Run detection
            annotated_frame, detections = self.detect_frame(frame)
            
            # Check violations
            violations = self.check_violations(detections)
            
            if violations:
                violation_count += len(violations)
                violation_frames.append(frame_count)
                
                # Save frame if violations detected
                if save_frames:
                    frame_dir = 'violation_frames'
                    os.makedirs(frame_dir, exist_ok=True)
                    frame_filename = os.path.join(frame_dir, f'frame_{frame_count:06d}.jpg')
                    cv2.imwrite(frame_filename, frame)
                else:
                    frame_filename = ''
                
                # Log violations
                for violation in violations:
                    self.log_violation(
                        violation,
                        worker_id=f'W{(frame_count % 100):03d}',
                        location='Construction Site - Zone 1',
                        frame_path=frame_filename
                    )
                    self.generate_alert(violation, f'W{(frame_count % 100):03d}', 'Zone 1')
            
            # Write annotated frame
            out.write(annotated_frame)
            
            # Progress update
            if processed_count % 50 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Processed {processed_count} frames, {violation_count} violations detected")
        
        cap.release()
        out.release()
        
        stats = {
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'total_violations': violation_count,
            'violation_frames': len(violation_frames),
            'avg_fps': fps,
            'output_video': output_path
        }
        
        print(f"\n✅ Processing complete!")
        print(f"  Total Frames: {frame_count}")
        print(f"  Processed Frames: {processed_count}")
        print(f"  Violations Detected: {violation_count}")
        print(f"  Frames with Violations: {len(violation_frames)}")
        print(f"  Output saved to: {output_path}")
        
        return stats
    
    def process_image(self, image_path, output_path='output_image.jpg'):
        """
        Process single image and detect PPE violations
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
            
        Returns:
            detections: List of detections
            violations: List of violations
            stats: Detection statistics
        """
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Run detection
        annotated_frame, detections = self.detect_frame(frame)
        
        # Check violations
        violations = self.check_violations(detections)
        
        # Get statistics
        stats = self.get_statistics(detections)
        
        # Save annotated image
        cv2.imwrite(output_path, annotated_frame)
        
        print(f"\n🖼️  Image processed: {image_path}")
        print(f"  Total Detections: {len(detections)}")
        print(f"  Persons: {stats['persons']}")
        print(f"  ✓ Hardhats: {stats['hardhats']}")
        print(f"  ✓ Masks: {stats['masks']}")
        print(f"  ✓ Safety Vests: {stats['safety_vests']}")
        print(f"  ✗ NO-Hardhats: {stats['no_hardhats']}")
        print(f"  ✗ NO-Masks: {stats['no_masks']}")
        print(f"  ✗ NO-Safety Vests: {stats['no_safety_vests']}")
        print(f"  Violations: {len(violations)}")
        
        if violations:
            print("\n⚠️  Violations Found:")
            for v in violations:
                print(f"    - {v['violation_type']} (Confidence: {v['confidence']*100:.1f}%, Severity: {v['severity']})")
        else:
            print("\n✓ No violations detected - All safety protocols followed!")
        
        return detections, violations, stats


# Example usage and testing
if __name__ == "__main__":
    print("="*60)
    print("PPE Detection System - Construction Site Safety")
    print("="*60)
    
    # Initialize detector
    detector = PPEDetector(model_path='best.pt', confidence_threshold=0.5)
    
    print("\n✓ Model loaded successfully!")
    print(f"  Model Classes: {len(detector.class_names)}")
    print(f"  Violation Classes: {[detector.class_names[i] for i in detector.violation_classes]}")
    print(f"  PPE Classes: {[detector.class_names[i] for i in detector.ppe_classes]}")
    
    print("\n" + "="*60)
    print("Ready to process images/videos!")
    print("="*60)
    print("\nUsage:")
    print("  detector.process_image('image.jpg')")
    print("  detector.process_video('video.mp4')")
    print("\nOr run: python run_detection.py --help")
    print("="*60)