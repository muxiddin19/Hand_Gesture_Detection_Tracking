import cv2
import mediapipe as mp
import numpy as np
import os
import json
import random
from pathlib import Path
from pycocotools.coco import COCO
import time
#import leap  # Leap Motion SDK
#from leap import Leap
import NatNetClient  # OptiTrack SDK
from scipy.spatial.transform import Rotation
from filterpy.kalman import KalmanFilter
import NatNetClient  # OptiTrack SDK

class handDetector():
    def __init__(self, static_mode=True, max_hands=2, model_complexity=1, 
                 detection_confidence=0.5, tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mpdraw = mp.solutions.drawing_utils

        self.adaptive_threshold = 0.5
        self.use_depth_hint = False
        self.last_imu_data = None

        # Leap Motion and OptiTrack initialization
        #self.controller = leap.Controller()
        self.natnet_client = NatNetClient.NatNetClient()
        #self.natnet_client.set_callback(self.optitrack_callback)
        self.natnet_client.new_frame_listener = self.optitrack_callback
        #self.natnet_client.set_new_framecallback(self.optitrack_callback)
        self.optitrack_data = None

        # Kalman filter for smoothing finger tip positions
        self.kalman_filters = {id: self.init_kalman_filter() for id in [12, 16, 20]}

    def init_kalman_filter(self):
        kf = KalmanFilter(dim_x=6, dim_z=3)  # State: [x, y, z, vx, vy, vz], Measurement: [x, y, z]
        kf.x = np.zeros(6)  # Initial state
        kf.F = np.array([[1, 0, 0, 1, 0, 0],  # State transition matrix
                         [0, 1, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0, 0, 0],  # Measurement matrix
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        kf.P *= 1e2  # Initial covariance
        kf.R = np.eye(3) * 5  # Measurement noise
        kf.Q = np.eye(6) * 0.1  # Process noise
        return kf

    def optitrack_callback(self, rigid_body_data):
        for rb in rigid_body_data:
            self.optitrack_data = rb  # {id, pos_x, pos_y, pos_z}

    def map_optitrack_to_mediapipe(self, landmarks):
        if not self.optitrack_data or not landmarks:
            return landmarks
        opti_pos = self.optitrack_data
        h, w, _ = (480, 640, 3)
        wrist_x, wrist_y = int(opti_pos.pos_x * w / 10), int(opti_pos.pos_y * h / 10)
        wrist_z = opti_pos.pos_z
        landmarks[0] = [0, wrist_x, wrist_y, wrist_z]
        return landmarks

    def fuse_leap_motion(self, landmarks):
        frame = self.controller.frame()
        if not frame.hands or len(landmarks) < 21:
            return landmarks

        leap_hand = frame.hands[0]
        leap_fingers = leap_hand.fingers
        finger_map = {1: 8, 2: 12, 3: 16, 4: 20}  # Leap finger indices to MediaPipe landmarks (index to pinky)
        for finger in leap_fingers:
            finger_idx = finger.type  # 1=index, 2=middle, 3=ring, 4=pinky
            if finger_idx in finger_map:
                tip_pos = finger.tip_position
                h, w, _ = (480, 640, 3)
                x, y = int(tip_pos.x * w / 200), int(tip_pos.y * h / 200)
                landmark_id = finger_map[finger_idx]
                landmarks[landmark_id] = [landmark_id, x, y, landmarks[landmark_id][3]]
        return landmarks

    def predict_occluded_finger(self, landmarks, finger_tip_id):
        # Use proximal joint (e.g., landmark 9 for middle finger) to predict tip (landmark 12)
        if finger_tip_id == 12:  # Middle finger
            base_id, mid_id = 9, 10  # Proximal and intermediate joints
        elif finger_tip_id == 16:  # Ring finger
            base_id, mid_id = 13, 14
        elif finger_tip_id == 20:  # Pinky finger
            base_id, mid_id = 17, 18
        else:
            return landmarks[finger_tip_id]

        if base_id >= len(landmarks) or mid_id >= len(landmarks):
            return landmarks[finger_tip_id]

        # Calculate vector from base to mid joint
        base = np.array(landmarks[base_id][1:4])  # [x, y, z]
        mid = np.array(landmarks[mid_id][1:4])
        vec = mid - base
        # Estimate tip position by extending the vector (assume tip is 1.5x the length)
        tip = mid + 1.5 * vec
        return [finger_tip_id, int(tip[0]), int(tip[1]), tip[2]]

    def smooth_finger_tip(self, landmark_id, landmark):
        kf = self.kalman_filters[landmark_id]
        kf.predict()
        measurement = np.array([landmark[1], landmark[2], landmark[3]], dtype=np.float32)
        kf.update(measurement)
        smoothed = kf.x[:3]  # Smoothed [x, y, z]
        return [landmark_id, int(smoothed[0]), int(smoothed[1]), smoothed[2]]

    def detect_typing_gesture(self, landmarks, threshold=0.05):
        if not landmarks or len(landmarks) < 21:
            return False
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        distance = np.sqrt((thumb_tip[1] - index_tip[1])**2 + (thumb_tip[2] - index_tip[2])**2)
        return distance < threshold

    def get_hand_location(self, frame, hand_num=0, draw_landmark=True):
        self.landmark_list = []
        is_typing = False
        if self.results.multi_hand_landmarks:
            hand = (self.results.multi_hand_landmarks[hand_num] 
                    if len(self.results.multi_hand_landmarks) > hand_num 
                    else self.results.multi_hand_landmarks[0])
            
            for id, landmark in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cz = landmark.z
                self.landmark_list.append([id, cx, cy, cz])
                if draw_landmark:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                    if id in [8, 12, 16, 20]:
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 0), cv2.FILLED)

            # Fuse Leap Motion data
            #self.landmark_list = self.fuse_leap_motion(self.landmark_list)
            # Use OptiTrack to anchor wrist position
            self.landmark_list = self.map_optitrack_to_mediapipe(self.landmark_list)

            # Predict occluded finger tips (middle, ring, pinky)
            for finger_tip_id in [12, 16, 20]:
                # If the finger tip's depth (z) is unreliable (e.g., too close to wrist), predict it
                if self.landmark_list[finger_tip_id][3] > -0.05:  # Arbitrary threshold for occlusion
                    self.landmark_list[finger_tip_id] = self.predict_occluded_finger(self.landmark_list, finger_tip_id)
                # Smooth the finger tip position
                self.landmark_list[finger_tip_id] = self.smooth_finger_tip(finger_tip_id, self.landmark_list[finger_tip_id])

            # Detect typing gesture
            is_typing = self.detect_typing_gesture(self.landmark_list)

        print(f"Detected landmarks: {len(self.landmark_list)}")
        return self.landmark_list, is_typing

    def find_hands(self, frame, draw=True):
        if frame is None or frame.size == 0:
            print("Error: Frame is None or empty")
            return frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(frame_rgb)
        print(f"Hand landmarks detected: {self.results.multi_hand_landmarks is not None}")
        return frame

def render_vr_keyboard(landmarks, is_typing):
    # Simplified OpenXR rendering (as in previous response)
    keyboard_layout = {
        'q': (0.1, 0.1), 'w': (0.2, 0.1), 'e': (0.3, 0.1),
        # Add more keys...
    }
    if landmarks and len(landmarks) >= 21:
        for finger_tip_id in [8, 12, 16, 20]:  # Index, middle, ring, pinky
            tip = landmarks[finger_tip_id]
            tip_x, tip_y = tip[1] / 640, tip[2] / 480
            closest_key = min(keyboard_layout.items(), key=lambda k: np.sqrt((tip_x - k[1][0])**2 + (tip_y - k[1][1])**2))
            key, (key_x, key_y) = closest_key
            if is_typing and finger_tip_id == 8:  # Use index finger for typing detection
                print(f"Key pressed: {key}")

def main():
    detector = handDetector(detection_confidence=0.5, tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)  # VR headset camera
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        frame_processed = detector.find_hands(frame, draw=False)
        landmarks, is_typing = detector.get_hand_location(frame_processed, draw_landmark=True)
        
        render_vr_keyboard(landmarks, is_typing)
        
        cv2.imshow("Frame", frame_processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()