import cv2
import mediapipe as mp
import time
import numpy as np
import imageio
from datetime import datetime
from collections import deque

class handDetector():
    def __init__(self, static_mode=False, max_hands=2, model_complexity=1, 
                 detection_confidence=0.5, tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe Hands
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mpdraw = mp.solutions.drawing_utils

        # Adaptive parameters
        self.adaptive_threshold = 0.5  # Initial confidence threshold for adaptation
        self.use_depth_hint = False    # Flag for depth-based enhancement
        self.last_imu_data = None      # Placeholder for VR IMU data
        
        # Metrics tracking
        self.detection_history = deque(maxlen=30)  # Track last 30 frames
        self.landmark_confidence = []
        self.processing_times = deque(maxlen=100)
        self.false_positives = 0
        self.false_negatives = 0
        self.true_positives = 0
        self.frame_count = 0

    def find_hands(self, frame, draw=True):
        """Detect hands and apply adaptive adjustments."""
        start_time = time.time()
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Simulate VR context (e.g., head movement affecting hand visibility)
        if self.last_imu_data:
            head_movement = self._estimate_occlusion_from_imu(self.last_imu_data)
            if head_movement > 0.7:  # High movement, lower confidence threshold
                self.adaptive_threshold = 0.3
            else:
                self.adaptive_threshold = 0.5

        # Process frame with adjusted confidence
        self.results = self.hands.process(frame_rgb)
        
        # Track detection success for metrics
        detected = self.results.multi_hand_landmarks is not None
        self.detection_history.append(1 if detected else 0)
        
        # If hands expected but not detected, count as false negative
        if not detected and self.frame_count > 5 and sum(self.detection_history) > 0.5 * len(self.detection_history):
            self.false_negatives += 1
            
        # Update processing time metrics
        process_time = time.time() - start_time
        self.processing_times.append(process_time)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    custom_drawing_spec = self.mpdraw.DrawingSpec(color=(0,0,0), thickness=0, circle_radius=0)
                    landmark_drawing_spec = self.mpdraw.DrawingSpec(color=(0, 0, 255), thickness=2)

                    self.mpdraw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mphands.HAND_CONNECTIONS,
                        custom_drawing_spec,  # This makes the landmark points invisible
                        landmark_drawing_spec  # This keeps the connections between points
                    )
                    
                # Track confidence of landmarks for metrics
                self.landmark_confidence = [lm.visibility if hasattr(lm, 'visibility') else 1.0 
                                           for lm in hand_landmarks.landmark]
                
                # If hand detected with high confidence, count as true positive
                if sum(self.landmark_confidence) / len(self.landmark_confidence) > 0.7:
                    self.true_positives += 1
                else:
                    # Low confidence detection might be a false positive
                    self.false_positives += 1
                
                # Apply depth hint if enabled
                if self.use_depth_hint:
                    frame = self._apply_depth_enhancement(frame, hand_landmarks)
        
        self.frame_count += 1
        return frame

    def get_hand_location(self, frame, hand_num=0, draw_landmark=True):
        """Get hand landmark locations with adaptive filtering."""
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = (self.results.multi_hand_landmarks[hand_num] 
                    if len(self.results.multi_hand_landmarks) > hand_num 
                    else self.results.multi_hand_landmarks[0])
            
            for id, landmark in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                # Filter low-confidence landmarks based on adaptive threshold
                if confidence >= self.adaptive_threshold:
                    self.landmark_list.append([id, cx, cy])
                    if draw_landmark:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.landmark_list

    def _estimate_occlusion_from_imu(self, imu_data):
        """Simulate occlusion estimation from VR IMU data (placeholder)."""
        # Placeholder: Use dummy IMU data (e.g., acceleration or orientation)
        if not self.last_imu_data:
            return 0.0
        # Simulate higher occlusion with rapid head movement
        accel_x = imu_data.get('accel_x', 0.0)
        return min(1.0, abs(accel_x) * 0.5)  # Simple heuristic

    def _apply_depth_enhancement(self, frame, hand_landmarks):
        """Placeholder for depth-based landmark refinement."""
        if not self.use_depth_hint:
            return frame
        # Simulate depth adjustment (e.g., shift landmarks based on depth)
        h, w, c = frame.shape
        for id, landmark in enumerate(hand_landmarks.landmark):
            if id in [8, 12, 16, 20]:  # Finger tips (index, middle, ring, pinky)
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                # Dummy depth adjustment (e.g., move closer if "near")
                depth_factor = 0.9 if id % 4 == 0 else 1.0  # Simple test
                cx = int(cx * depth_factor)
        return frame

    def update_context(self, imu_data=None, depth_hint=None):
        """Update adaptive context with VR data."""
        self.last_imu_data = imu_data or {'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0}
        self.use_depth_hint = depth_hint is not None
        
    def get_metrics(self):
        """Calculate and return performance metrics."""
        if self.frame_count == 0:
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "fps": 0,
                "avg_landmark_confidence": 0,
                "detection_rate": 0
            }
            
        # Calculate metrics
        total_attempts = self.true_positives + self.false_positives + self.false_negatives
        accuracy = self.true_positives / max(1, total_attempts) * 100
        precision = self.true_positives / max(1, (self.true_positives + self.false_positives)) * 100
        recall = self.true_positives / max(1, (self.true_positives + self.false_negatives)) * 100
        avg_fps = 1.0 / (sum(self.processing_times) / max(1, len(self.processing_times)))
        avg_confidence = sum(self.landmark_confidence) / max(1, len(self.landmark_confidence)) * 100
        detection_rate = sum(self.detection_history) / max(1, len(self.detection_history)) * 100
        
        return {
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "fps": round(avg_fps, 1),
            "avg_landmark_confidence": round(avg_confidence, 2),
            "detection_rate": round(detection_rate, 2)
        }

# Example usage
def main():
    # GIF parameters
    gif_duration = 5  # seconds
    target_fps = 15  # Lower for smaller file size
    frames = []
    max_frames = gif_duration * target_fps
    
    # Metrics logging
    metrics_log = []
    
    # Resolution settings (lower for smaller GIF)
    width, height = 640, 480
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    detector = handDetector(detection_confidence=0.7, tracking_confidence=0.7)
    
    start_time = time.time()
    frame_count = 0
    
    while len(frames) < max_frames:
        success, frame = cap.read()
        if not success:
            break
            
        # Resize for smaller file size if needed
        frame = cv2.resize(frame, (width, height))
        
        # Simulate VR IMU data (replace with real data from VR SDK later)
        imu_data = {'accel_x': np.sin(time.time()), 'accel_y': 0.1, 'accel_z': 0.0}
        detector.update_context(imu_data=imu_data, depth_hint=True)

        # Process frame
        frame = detector.find_hands(frame)
        landmarks = detector.get_hand_location(frame)
        
        # Calculate metrics every 15 frames
        if frame_count % 15 == 0:
            metrics = detector.get_metrics()
            metrics_log.append(metrics)
            
            # Display metrics on frame
            cv2.putText(frame, f"Accuracy: {metrics['accuracy']}%", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {metrics['fps']}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {metrics['avg_landmark_confidence']}%", (10, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add frame to GIF (convert BGR to RGB for imageio)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb_frame)
        
        # Display real-time view
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        frame_count += 1
        
        # Control frame rate for GIF
        elapsed = time.time() - start_time
        expected_frames = elapsed * target_fps
        if frame_count > expected_frames:
            time.sleep(0.01)  # Small delay to maintain target FPS
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save GIF with compression settings to keep under 10MB
    print("Saving GIF...")
    imageio.mimsave(f"hand_tracking_{timestamp}.gif", frames, fps=target_fps, optimize=True, duration=1000/target_fps)
    
    # Save metrics to CSV
    import csv
    with open(f"metrics_{timestamp}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics_log[0].keys())
        writer.writeheader()
        writer.writerows(metrics_log)
    
    # Calculate and save final metrics summary
    final_metrics = detector.get_metrics()
    print("\nFinal Performance Metrics:")
    print(f"Accuracy: {final_metrics['accuracy']}%")
    print(f"Precision: {final_metrics['precision']}%")
    print(f"Recall: {final_metrics['recall']}%")
    print(f"Average FPS: {final_metrics['fps']}")
    print(f"Average Landmark Confidence: {final_metrics['avg_landmark_confidence']}%")
    print(f"Detection Rate: {final_metrics['detection_rate']}%")
    print(f"Total Frames Processed: {detector.frame_count}")
    print(f"GIF saved as: hand_tracking_{timestamp}.gif")
    print(f"Metrics saved as: metrics_{timestamp}.csv")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()