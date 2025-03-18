# Description: This script is used to detect hands in images using MediaPipe Hands model. It also evaluates the performance of the hand detection model on a dataset of hand images. The script uses the COCO dataset format for annotations and images. The script uses the COCO API to load annotations and images from the dataset. The script evaluates the hand detection model by comparing the detected landmarks with the ground truth landmarks in the dataset. The script also applies occlusion and lighting changes to the images to simulate real-world conditions. The script uses the IMU data and depth hint to adapt the detection threshold based on the head movement and depth information. The script prints the average landmark error and FPS for the baseline and adaptive features.
import cv2
import mediapipe as mp
import numpy as np
import os
import json
import random
from pathlib import Path
from pycocotools.coco import COCO
import time

class handDetector():
    def __init__(self, static_mode=False, max_hands=2, model_complexity=1, 
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

    def find_hands(self, frame, draw=True):
        if frame is None or frame.size == 0:
            print("Error: Frame is None or empty")
            return frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.last_imu_data:
            head_movement = self._estimate_occlusion_from_imu(self.last_imu_data)
            if head_movement > 0.7:
                self.adaptive_threshold = 0.3
            else:
                self.adaptive_threshold = 0.5

        self.results = self.hands.process(frame_rgb)
        print(f"Hand landmarks detected: {self.results.multi_hand_landmarks is not None}")

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(frame, hand_landmarks, self.mphands.HAND_CONNECTIONS)
                if self.use_depth_hint:
                    frame = self._apply_depth_enhancement(frame, hand_landmarks)
        return frame

    def get_hand_location(self, frame, hand_num=0, draw_landmark=True):
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            hand = (self.results.multi_hand_landmarks[hand_num] 
                    if len(self.results.multi_hand_landmarks) > hand_num 
                    else self.results.multi_hand_landmarks[0])
            
            for id, landmark in enumerate(hand.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                confidence = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                if confidence >= self.adaptive_threshold:
                    self.landmark_list.append([id, cx, cy])
                    if draw_landmark:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        print(f"Detected landmarks: {len(self.landmark_list)}")
        return self.landmark_list

    def _estimate_occlusion_from_imu(self, imu_data):
        accel_x = imu_data.get('accel_x', 0.0)
        return min(1.0, abs(accel_x) * 0.5)

    def _apply_depth_enhancement(self, frame, hand_landmarks):
        if not self.use_depth_hint:
            return frame
        h, w, c = frame.shape
        for id, landmark in enumerate(hand_landmarks.landmark):
            if id in [8, 12, 16, 20]:
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                depth_factor = 0.9 if random.random() > 0.5 else 1.0
                cx = int(cx * depth_factor)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        return frame

    def update_context(self, imu_data=None, depth_hint=None):
        self.last_imu_data = imu_data or {'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0}
        self.use_depth_hint = depth_hint is not None

def augment_image(frame, occlusion=True, lighting_change=True):
    frame = frame.copy()
    h, w, c = frame.shape

    if occlusion and random.random() > 0.5:
        x, y = random.randint(0, w//2), random.randint(0, h//2)
        frame[y:y+h//4, x:x+w//4] = 0

    if lighting_change and random.random() > 0.5:
        brightness = random.uniform(0.5, 1.5)
        frame = cv2.convertScaleAbs(frame, alpha=brightness, beta=0)

    return frame

def evaluate_hand_tracking(detector, dataset_path, split="train", num_samples=100):
    dataset_path = Path(dataset_path)
    annotation_path = dataset_path / f"coco_annotation/{split}/_annotations.coco.json"
    print(f"Loading COCO annotations from: {annotation_path}")
    if not annotation_path.exists():
        print(f"Error: Annotation file not found at {annotation_path}")
        return float('inf'), float('nan')

    try:
        coco = COCO(str(annotation_path))
    except Exception as e:
        print(f"Error loading COCO annotations: {e}")
        return float('inf'), float('nan')

    img_ids = coco.getImgIds()[:num_samples]
    print(f"Found {len(img_ids)} images in {split} split")
    if not img_ids:
        print("Error: No image IDs found in the dataset")
        return float('inf'), float('nan')

    total_error = 0
    total_landmarks = 0
    fps_list = []

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        img_path = dataset_path / f"images/{split}" / img_info['file_name']
        print(f"Loading image: {img_path}")
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Failed to load image: {img_path}")
            continue

        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if not anns:
            print(f"No annotations for image ID {img_id}")
            continue
        gt_landmarks = anns[0]['keypoints']
        gt_landmarks = [(gt_landmarks[i], gt_landmarks[i+1]) for i in range(0, len(gt_landmarks), 3) if gt_landmarks[i+2] > 0]
        print(f"Ground truth landmarks: {len(gt_landmarks)}")

        frame_augmented = augment_image(frame)
        start_time = time.time()
        frame_processed = detector.find_hands(frame_augmented, draw=False)
        detected_landmarks = detector.get_hand_location(frame_processed, draw_landmark=False)
        fps = 1 / (time.time() - start_time)
        fps_list.append(fps)
        print(f"FPS for image {img_id}: {fps}")

        if detected_landmarks and gt_landmarks:
            print(f"Image {img_id}: Detected {len(detected_landmarks)}, GT {len(gt_landmarks)}")
            for det_lm in detected_landmarks:
                id, det_x, det_y = det_lm
                if id < len(gt_landmarks):
                    gt_x, gt_y = gt_landmarks[id]
                    error = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2)
                    total_error += error
                    total_landmarks += 1
        else:
            print(f"Image {img_id}: No valid landmarks detected")

    avg_error = total_error / total_landmarks if total_landmarks > 0 else float('inf')
    avg_fps = np.mean(fps_list) if fps_list else float('nan')
    return avg_error, avg_fps

def main():
    detector = handDetector(detection_confidence=0.5, tracking_confidence=0.5)

    # Test a single image to confirm MediaPipe detection
    test_image_path = "C:/users/muhiddin/.cache/kagglehub/datasets/riondsilva21/hand-keypoint-dataset-26k/versions/3/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k/images/train/0.jpg"
    print(f"Testing MediaPipe on a single image: {test_image_path}")
    test_frame = cv2.imread(test_image_path)
    if test_frame is not None:
        detector.find_hands(test_frame, draw=False)
        detector.get_hand_location(test_frame, draw_landmark=False)
    else:
        print(f"Failed to load test image: {test_image_path}")

    # Evaluate on dataset
    dataset_path = Path("C:/users/muhiddin/.cache/kagglehub/datasets/riondsilva21/hand-keypoint-dataset-26k/versions/3/hand_keypoint_dataset_26k/hand_keypoint_dataset_26k")
    print("Evaluating baseline performance...")
    baseline_error, baseline_fps = evaluate_hand_tracking(detector, dataset_path, split="train", num_samples=10)  # Reduced for faster debugging

    detector.update_context(imu_data={'accel_x': 0.8}, depth_hint=True)
    print("Evaluating with adaptive features...")
    adaptive_error, adaptive_fps = evaluate_hand_tracking(detector, dataset_path, split="train", num_samples=10)

    print(f"Baseline - Avg Landmark Error: {baseline_error:.2f} pixels, FPS: {baseline_fps:.2f}")
    print(f"Adaptive - Avg Landmark Error: {adaptive_error:.2f} pixels, FPS: {adaptive_fps:.2f}")

if __name__ == "__main__":
    main()