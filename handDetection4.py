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
        
        # Frame data storage for temporal consistency (last N frames)
        self.frame_buffer_size = 5
        self.previous_frames = []
        self.previous_landmarks = []
        
        # Metrics tracking
        self.processing_times = deque(maxlen=100)  # Processing time history
        self.frame_count = 0  # Total frames processed
        
        # Detection statistics
        self.landmark_history = []  # Store landmark data for each frame
        self.landmark_confidences = []  # Store confidences for each landmark
        self.detection_count = 0  # Number of frames where hands were detected
        self.continuity_breaks = 0  # Number of times tracking was lost
        self.was_hand_in_prev_frame = False  # For tracking continuity
        
        # Temporal consistency metrics
        self.landmark_displacements = []  # Track movement of landmarks between frames
        self.expected_vs_actual = []  # Compare predicted vs actual positions

    def find_hands(self, frame, draw=True):
        """Detect hands and apply adaptive adjustments."""
        start_time = time.time()
        frame_copy = frame.copy()
        
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
        
        # Update processing time metrics
        process_time = time.time() - start_time
        self.processing_times.append(process_time)
        
        # Track detection continuity
        current_frame_has_hand = self.results.multi_hand_landmarks is not None
        if self.was_hand_in_prev_frame and not current_frame_has_hand:
            self.continuity_breaks += 1
        self.was_hand_in_prev_frame = current_frame_has_hand
        
        # Update detection count
        if current_frame_has_hand:
            self.detection_count += 1

        # Current frame landmarks
        current_landmarks = []
        current_confidences = []
        
        if self.results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                if draw:
                    custom_drawing_spec = self.mpdraw.DrawingSpec(color=(0,0,0), thickness=0, circle_radius=0)
                    landmark_drawing_spec = self.mpdraw.DrawingSpec(color=(0, 0, 255), thickness=2)

                    self.mpdraw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mphands.HAND_CONNECTIONS,
                        custom_drawing_spec,
                        landmark_drawing_spec
                    )
                
                # Extract landmark positions and confidences for metrics
                h, w, c = frame.shape
                hand_landmarks_data = []
                hand_confidences = []
                
                for lm_idx, landmark in enumerate(hand_landmarks.landmark):
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                    
                    hand_landmarks_data.append((cx, cy))
                    hand_confidences.append(confidence)
                    
                    # Draw visible landmarks based on confidence
                    if draw and confidence >= self.adaptive_threshold:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                current_landmarks.append(hand_landmarks_data)
                current_confidences.extend(hand_confidences)
                
                # Apply depth hint if enabled
                if self.use_depth_hint:
                    frame = self._apply_depth_enhancement(frame, hand_landmarks)
        
        # Calculate temporal consistency if we have previous data
        if current_landmarks and self.previous_landmarks:
            self._calculate_temporal_metrics(current_landmarks)
        
        # Store current frame data
        self.landmark_history.append(current_landmarks)
        if current_confidences:
            self.landmark_confidences.extend(current_confidences)
            
        # Keep only the most recent frames
        if len(self.landmark_history) > self.frame_buffer_size:
            self.landmark_history.pop(0)
            
        # Store for next frame comparison
        self.previous_landmarks = current_landmarks
        self.previous_frames.append(frame_copy)
        if len(self.previous_frames) > self.frame_buffer_size:
            self.previous_frames.pop(0)
        
        self.frame_count += 1
        return frame

    def _calculate_temporal_metrics(self, current_landmarks):
        """Calculate metrics related to temporal consistency of tracking."""
        # Only compare if there's at least one hand in current and previous frame
        if not current_landmarks or not self.previous_landmarks:
            return
            
        # For simplicity, just compare the first hand in both frames
        # In a real research implementation, you'd match hands across frames
        try:
            prev_hand = self.previous_landmarks[0]
            curr_hand = current_landmarks[0]
            
            # Calculate displacements between corresponding landmarks
            if len(prev_hand) == len(curr_hand):
                for i in range(len(prev_hand)):
                    prev_x, prev_y = prev_hand[i]
                    curr_x, curr_y = curr_hand[i]
                    
                    # Euclidean distance between successive positions
                    displacement = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
                    self.landmark_displacements.append(displacement)
                    
                    # In a real implementation, you'd have motion prediction here
                    # For now, simple expected position based on previous movement
                    if len(self.landmark_history) >= 3:
                        prev_prev_hand = self.landmark_history[-3][0]
                        if i < len(prev_prev_hand):
                            # Simple linear prediction
                            prev_prev_x, prev_prev_y = prev_prev_hand[i]
                            expected_x = prev_x + (prev_x - prev_prev_x)
                            expected_y = prev_y + (prev_y - prev_prev_y)
                            
                            # Error between prediction and actual
                            prediction_error = np.sqrt((curr_x - expected_x)**2 + (curr_y - expected_y)**2)
                            self.expected_vs_actual.append(prediction_error)
        except (IndexError, ValueError):
            # Handle cases where hand detection might change between frames
            pass

    def get_hand_location(self, frame, hand_num=0, draw_landmark=False):
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
        if not self.last_imu_data:
            return 0.0
        accel_x = imu_data.get('accel_x', 0.0)
        return min(1.0, abs(accel_x) * 0.5)

    def _apply_depth_enhancement(self, frame, hand_landmarks):
        """Placeholder for depth-based landmark refinement."""
        if not self.use_depth_hint:
            return frame
        h, w, c = frame.shape
        for id, landmark in enumerate(hand_landmarks.landmark):
            if id in [8, 12, 16, 20]:  # Finger tips (index, middle, ring, pinky)
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                depth_factor = 0.9 if id % 4 == 0 else 1.0
                cx = int(cx * depth_factor)
        return frame

    def update_context(self, imu_data=None, depth_hint=None):
        """Update adaptive context with VR data."""
        self.last_imu_data = imu_data or {'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0}
        self.use_depth_hint = depth_hint is not None
        
    def get_metrics(self):
        """Calculate and return comprehensive research metrics.
        
        This uses actual calculations based on the collected data.
        """
        if self.frame_count == 0:
            return {
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
                "fps": 0,
                "avg_landmark_confidence": 0,
                "detection_rate": 0,
                "jitter": 0,
                "prediction_accuracy": 0
            }
            
        # Calculate true metrics from collected data:
        
        # 1. Detection Rate (percentage of frames where hands were detected)
        detection_rate = (self.detection_count / self.frame_count) * 100
        
        # 2. Landmark Confidence (average confidence score of all landmarks)
        avg_landmark_confidence = 0
        if self.landmark_confidences:
            avg_landmark_confidence = (sum(self.landmark_confidences) / len(self.landmark_confidences)) * 100
            
        # 3. Accuracy calculation (percentage of high-confidence landmarks)
        # In real research, accuracy would compare against ground truth
        # Here we use the proportion of landmarks above confidence threshold
        high_confidence_count = sum(1 for conf in self.landmark_confidences if conf >= self.adaptive_threshold)
        total_landmarks = len(self.landmark_confidences) if self.landmark_confidences else 1
        accuracy = (high_confidence_count / total_landmarks) * 100 if self.landmark_confidences else 0
            
        # 4. Jitter (average displacement between successive frames)
        # Lower values indicate more stable tracking
        jitter = 0
        if self.landmark_displacements:
            jitter = sum(self.landmark_displacements) / len(self.landmark_displacements)
            
        # 5. Prediction Accuracy (how close predicted positions are to actual)
        # Lower values indicate better prediction
        prediction_accuracy = 0
        if self.expected_vs_actual:
            prediction_accuracy = sum(self.expected_vs_actual) / len(self.expected_vs_actual)
            # Convert to percentage where 100% is perfect prediction (0 error)
            # Using an exponential decay function to map pixel errors to percentage
            # 0 pixels error = 100%, 10 pixels error ≈ 90%, 30 pixels error ≈ 74%, etc.
            prediction_accuracy = 100 * np.exp(-0.01 * prediction_accuracy)
            
        # 6. Tracking Stability (inverse of number of times tracking was lost)
        # Higher values indicate more stable tracking across frames
        continuity = max(0, 100 - (self.continuity_breaks / max(1, self.frame_count) * 100))
        
        # 7. Precision (in research context: confidence in detected landmarks)
        # Using proportion of landmarks above high confidence threshold (0.8)
        very_high_confidence = sum(1 for conf in self.landmark_confidences if conf >= 0.8)
        precision = (very_high_confidence / total_landmarks) * 100 if self.landmark_confidences else 0
        
        # 8. Recall (in research context: proportion of expected landmarks actually detected)
        # MediaPipe hand model has 21 landmarks per hand, calculate what percentage were detected
        expected_landmarks = 21 * self.detection_count  # 21 landmarks per hand per detection
        actual_landmarks = sum(len(hand) for frame_landmarks in self.landmark_history 
                               for hand in frame_landmarks) if self.landmark_history else 0
        recall = min(100, (actual_landmarks / max(1, expected_landmarks)) * 100)
        
        # 9. F1 Score (harmonic mean of precision and recall)
        f1_score = 0
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
            
        # 10. Frames Per Second
        avg_fps = 1.0 / (sum(self.processing_times) / max(1, len(self.processing_times)))
        
        return {
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "f1_score": round(f1_score, 2),
            "fps": round(avg_fps, 1),
            "avg_landmark_confidence": round(avg_landmark_confidence, 2),
            "detection_rate": round(detection_rate, 2),
            "jitter": round(jitter, 2),
            "prediction_accuracy": round(prediction_accuracy, 2),
            "stability": round(continuity, 2)
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
    
    detector = handDetector(detection_confidence=0.5, tracking_confidence=0.5)
    
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
            cv2.putText(frame, f"Precision: {metrics['precision']}%", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Recall: {metrics['recall']}%", (10, 90), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"F1: {metrics['f1_score']}%", (10, 120), 
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
    print(f"F1 Score: {final_metrics['f1_score']}%")
    print(f"Average FPS: {final_metrics['fps']}")
    print(f"Average Landmark Confidence: {final_metrics['avg_landmark_confidence']}%")
    print(f"Detection Rate: {final_metrics['detection_rate']}%")
    print(f"Jitter: {final_metrics['jitter']} pixels")
    print(f"Prediction Accuracy: {final_metrics['prediction_accuracy']}%")
    print(f"Tracking Stability: {final_metrics['stability']}%")
    print(f"Total Frames Processed: {detector.frame_count}")
    print(f"Frames with Hand Detections: {detector.detection_count}")
    print(f"GIF saved as: hand_tracking_{timestamp}.gif")
    print(f"Metrics saved as: metrics_{timestamp}.csv")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()