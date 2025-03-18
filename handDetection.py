import cv2
import mediapipe as mp
import time
import numpy as np

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

    def find_hands(self, frame, draw=True):
        """Detect hands and apply adaptive adjustments."""
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

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    #self.mpdraw.draw_landmarks(frame, hand_landmarks, self.mphands.HAND_CONNECTIONS)
                    # Add this to your code
                    custom_drawing_spec = self.mpdraw.DrawingSpec(color=(0,0,0), thickness=0, circle_radius=0)
                    landmark_drawing_spec = self.mpdraw.DrawingSpec(color=(0, 0, 255), thickness=2)

                    # Then replace the original drawing call with this
                    self.mpdraw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mphands.HAND_CONNECTIONS,
                        custom_drawing_spec,  # This makes the landmark points invisible
                        landmark_drawing_spec  # This keeps the connections between points
                    )
                # Apply depth hint if enabled (placeholder for now)
                if self.use_depth_hint:
                    frame = self._apply_depth_enhancement(frame, hand_landmarks)
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
                #cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)  # Green for depth-adjusted
        return frame

    def update_context(self, imu_data=None, depth_hint=None):
        """Update adaptive context with VR data."""
        self.last_imu_data = imu_data or {'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0}
        self.use_depth_hint = depth_hint is not None

# Example usage
def main():
    cap = cv2.VideoCapture(0)  # Use webcam or replace with VR camera feed
    detector = handDetector(detection_confidence=0.7, tracking_confidence=0.7)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Simulate VR IMU data (replace with real data from VR SDK later)
        imu_data = {'accel_x': np.sin(time.time()), 'accel_y': 0.1, 'accel_z': 0.0}
        detector.update_context(imu_data=imu_data, depth_hint=True)

        # Process frame
        frame = detector.find_hands(frame)
        landmarks = detector.get_hand_location(frame)

        # Display results
        cv2.imshow("Hand Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()