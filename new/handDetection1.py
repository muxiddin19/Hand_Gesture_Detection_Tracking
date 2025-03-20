import cv2
import mediapipe as mp
import numpy as np
import time
from filterpy.kalman import KalmanFilter
import json
import socket
import threading
import sys
import socket
import json
import threading
import time

# VR Integration imports
try:
    import triad_openvr  # For OpenVR/SteamVR integration
except ImportError:
    print("SteamVR integration not available. Install with: pip install triad_openvr")
    
try:
    import optitrack.natnet as natnet  # For OptiTrack integration
except ImportError:
    print("OptiTrack integration not available. Install with: pip install optitrack-natnet")
    
try:
    from leapuvc import LeapUVC  # For Leap Motion integration
except ImportError:
    print("Leap Motion integration not available. Install with: pip install leapuvc")

class HandTrackingSystem:
    def __init__(self, tracking_system="leap", unity_integration=True, unity_port=12345):
        """
        Initialize Hand Tracking System for VR
        
        Args:
            tracking_system (str): Choice of tracking system - "leap", "optitrack", or "steamvr"
            unity_integration (bool): Whether to stream data to Unity
            unity_port (int): Port number for Unity communication
        """
        self.tracking_system = tracking_system.lower()
        self.vr_initialized = False
        self.unity_integration = unity_integration
        self.unity_port = unity_port
        self.unity_socket = None
        self.running = False
        
        # Initialize tracking systems
        self.init_tracking_system()
        
        # Fallback to MediaPipe if tracking initialization fails
        if not self.vr_initialized:
            print("Falling back to camera-based tracking with MediaPipe")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
        
        # Set up Kalman filters for smooth tracking
        self.setup_kalman_filters()
        
        # Initialize Unity communication if enabled
        if self.unity_integration:
            self.init_unity_connection()
            self.unity_integration = unity_integration
            self.unity_address = unity_address
            self.unity_socket = None
            self.setup_unity_connection()
    
    def setup_unity_connection(self):
        """Initialize connection to Unity"""
        if self.unity_integration:
            try:
                self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                print(f"Unity integration enabled. Will send data to {self.unity_address}")
            except Exception as e:
                print(f"Failed to initialize Unity connection: {e}")
                self.unity_integration = False
    
    def send_to_unity(self, tracking_data):
        """Send hand tracking data to Unity"""
        if not self.unity_integration or not self.unity_socket:
            return
        
        try:
            # Format data for Unity
            unity_data = self.format_data_for_unity(tracking_data)
            
            # Convert to JSON and send
            json_data = json.dumps(unity_data).encode('utf-8')
            self.unity_socket.sendto(json_data, self.unity_address)
        except Exception as e:
            print(f"Error sending data to Unity: {e}")
    
    def format_data_for_unity(self, tracking_data):
        """Format tracking data specifically for Unity consumption"""
        unity_data = {
            "timestamp": time.time(),
            "hands": []
        }
        
        # Process left hand
        if tracking_data["left_hand"]["present"]:
            left_hand = {
                "type": "left",
                "landmarks": [],
                "gesture": tracking_data["left_hand"].get("gesture", "none"),
                "confidence": tracking_data["left_hand"].get("confidence", 0.0)
            }
            
            # Add normalized 3D positions for each landmark
            for lm in tracking_data["left_hand"]["landmarks"]:
                left_hand["landmarks"].append({
                    "x": lm[0],  # x: 0-1 normalized
                    "y": lm[1],  # y: 0-1 normalized  
                    "z": lm[2]   # z: depth
                })
            
            unity_data["hands"].append(left_hand)
        
        # Process right hand
        if tracking_data["right_hand"]["present"]:
            right_hand = {
                "type": "right",
                "landmarks": [],
                "gesture": tracking_data["right_hand"].get("gesture", "none"),
                "confidence": tracking_data["right_hand"].get("confidence", 0.0)
            }
            
            # Add normalized 3D positions for each landmark
            for lm in tracking_data["right_hand"]["landmarks"]:
                right_hand["landmarks"].append({
                    "x": lm[0],
                    "y": lm[1],
                    "z": lm[2]
                })
            
            unity_data["hands"].append(right_hand)
            
        return unity_data
    
    def run_camera_mode(self):
        """Run in camera-based fallback mode using MediaPipe"""
        cap = cv2.VideoCapture(0)
        
        try:
            while self.running:
                success, img = cap.read()
                if not success:
                    continue
                
                # Flip image horizontally for selfie-view
                img = cv2.flip(img, 1)
                
                # Create a copy for visualization
                display_img = img.copy()
                
                # Process hand tracking directly with MediaPipe
                # Convert BGR to RGB for MediaPipe
                results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # Basic visualization from MediaPipe
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw native MediaPipe visualization
                        self.mp_draw.draw_landmarks(
                            display_img,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                
                # Process for our tracking data structure
                tracking_data = self.get_mediapipe_data(img)
                
                # Send data to Unity if enabled
                if self.unity_integration:
                    self.send_to_unity(tracking_data)
                
                # Additional custom visualization
                self.draw_debug_visualization(display_img, tracking_data)
                
                # Display
                cv2.imshow("Hand Tracking (Camera Mode)", display_img)
                
                # Break loop with 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error in camera mode: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.unity_socket:
                self.unity_socket.close()
    
    def init_tracking_system(self):
        """Initialize the chosen tracking system"""
        try:
            if self.tracking_system == "leap":
                # Initialize Leap Motion
                self.leap = LeapUVC()
                self.leap.open()
                self.vr_initialized = True
                print("Leap Motion tracking initialized")
                
            elif self.tracking_system == "optitrack":
                # Initialize OptiTrack
                self.optitrack_client = natnet.Client.connect(server="127.0.0.1")
                self.vr_initialized = True
                print("OptiTrack tracking initialized")
                
            elif self.tracking_system == "steamvr":
                # Initialize SteamVR/OpenVR
                self.vr = triad_openvr.triad_openvr()
                self.vr_initialized = True
                print("SteamVR tracking initialized")
                
            else:
                print(f"Unknown tracking system: {self.tracking_system}")
                return False
                
        except Exception as e:
            print(f"Failed to initialize {self.tracking_system}: {e}")
            self.vr_initialized = False
            return False
            
        return True
        
    # def init_unity_connection(self):
    #     """Initialize socket connection to Unity"""
    #     try:
    #         self.unity_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #         print(f"Unity communication initialized on port {self.unity_port}")
    #     except Exception as e:
    #         print(f"Failed to initialize Unity connection: {e}")
    #         self.unity_integration = False
    
    def setup_kalman_filters(self):
        """Set up Kalman filters for hand tracking"""
        # Create Kalman filters for each finger joint
        self.finger_filters = {}
        
        for hand in ["left", "right"]:
            self.finger_filters[hand] = {}
            for finger in ["thumb", "index", "middle", "ring", "pinky"]:
                self.finger_filters[hand][finger] = {}
                for joint in ["metacarpal", "proximal", "intermediate", "distal", "tip"]:
                    # Create a Kalman filter for each joint (x, y, z)
                    kf = KalmanFilter(dim_x=6, dim_z=3)  # 3D position and velocity
                    
                    # State transition matrix (position and velocity)
                    kf.F = np.array([
                        [1, 0, 0, 1, 0, 0],  # x position
                        [0, 1, 0, 0, 1, 0],  # y position
                        [0, 0, 1, 0, 0, 1],  # z position
                        [0, 0, 0, 1, 0, 0],  # x velocity
                        [0, 0, 0, 0, 1, 0],  # y velocity
                        [0, 0, 0, 0, 0, 1]   # z velocity
                    ])
                    
                    # Measurement function (position only)
                    kf.H = np.array([
                        [1, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0]
                    ])
                    
                    # Initial covariance matrix
                    kf.P *= 100
                    
                    # Measurement noise
                    kf.R = np.eye(3) * 0.01
                    
                    # Process noise
                    kf.Q = np.eye(6) * 0.01
                    
                    # Store the filter and initial state
                    self.finger_filters[hand][finger][joint] = {
                        "filter": kf,
                        "position": np.zeros(3),
                        "last_update": time.time()
                    }
        
        # Palm tracking
        self.palm_filters = {}
        for hand in ["left", "right"]:
            kf = KalmanFilter(dim_x=6, dim_z=3)
            kf.F = np.array([
                [1, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            kf.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0]
            ])
            kf.P *= 100
            kf.R = np.eye(3) * 0.01
            kf.Q = np.eye(6) * 0.01
            
            self.palm_filters[hand] = {
                "filter": kf,
                "position": np.zeros(3),
                "rotation": np.zeros(4),  # Quaternion
                "last_update": time.time()
            }
    
    def update_finger_position(self, hand, finger, joint, pos):
        """
        Update Kalman filter with new position measurement
        
        Args:
            hand: "left" or "right"
            finger: "thumb", "index", "middle", "ring", or "pinky"
            joint: "metacarpal", "proximal", "intermediate", "distal", or "tip"
            pos: (x, y, z) position
        
        Returns:
            Filtered position (x, y, z)
        """
        if hand not in self.finger_filters or finger not in self.finger_filters[hand] or joint not in self.finger_filters[hand][finger]:
            return pos
        
        filter_data = self.finger_filters[hand][finger][joint]
        kf = filter_data["filter"]
        
        # Calculate time since last update for velocity estimation
        current_time = time.time()
        dt = current_time - filter_data["last_update"]
        filter_data["last_update"] = current_time
        
        # Predict and update
        kf.predict()
        kf.update(np.array(pos))
        
        # Get filtered position
        filtered_state = kf.x
        filtered_pos = filtered_state[0:3]
        
        # Store the filtered position
        filter_data["position"] = filtered_pos
        
        return filtered_pos
    
    def update_palm_position(self, hand, pos, rot=None):
        """
        Update palm position and rotation
        
        Args:
            hand: "left" or "right"
            pos: (x, y, z) position
            rot: (x, y, z, w) quaternion rotation (optional)
        
        Returns:
            Filtered position (x, y, z)
        """
        if hand not in self.palm_filters:
            return pos
        
        filter_data = self.palm_filters[hand]
        kf = filter_data["filter"]
        
        # Calculate time since last update
        current_time = time.time()
        dt = current_time - filter_data["last_update"]
        filter_data["last_update"] = current_time
        
        # Predict and update position
        kf.predict()
        kf.update(np.array(pos))
        
        # Get filtered position
        filtered_state = kf.x
        filtered_pos = filtered_state[0:3]
        
        # Store the filtered position
        filter_data["position"] = filtered_pos
        
        # Store rotation if provided
        if rot is not None:
            filter_data["rotation"] = rot
        
        return filtered_pos
    
    def get_leap_motion_data(self):
        """
        Get hand tracking data from Leap Motion
        
        Returns:
            Dictionary containing hand tracking data
        """
        tracking_data = {
            "left_hand": {"present": False},
            "right_hand": {"present": False}
        }
        
        try:
            frame = self.leap.frame()
            
            if frame and frame.hands:
                for hand in frame.hands:
                    hand_key = "left_hand" if hand.is_left else "right_hand"
                    hand_data = {"present": True}
                    
                    # Get palm data
                    palm_pos = hand.palm_position
                    palm_rot = hand.palm_orientation  # Quaternion
                    
                    # Filter palm position
                    filtered_palm_pos = self.update_palm_position(
                        "left" if hand.is_left else "right",
                        palm_pos,
                        palm_rot
                    )
                    
                    hand_data["palm"] = {
                        "position": filtered_palm_pos.tolist(),
                        "rotation": palm_rot.tolist() if palm_rot is not None else [0, 0, 0, 1]
                    }
                    
                    # Get finger data
                    hand_data["fingers"] = {}
                    
                    for i, finger in enumerate(hand.fingers):
                        finger_name = ["thumb", "index", "middle", "ring", "pinky"][i]
                        finger_data = {}
                        
                        # Get joint positions
                        for j, joint_name in enumerate(["metacarpal", "proximal", "intermediate", "distal", "tip"]):
                            joint_pos = finger.bone(j).next_joint
                            
                            # Filter joint position
                            filtered_joint_pos = self.update_finger_position(
                                "left" if hand.is_left else "right",
                                finger_name,
                                joint_name,
                                joint_pos
                            )
                            
                            finger_data[joint_name] = {
                                "position": filtered_joint_pos.tolist()
                            }
                        
                        hand_data["fingers"][finger_name] = finger_data
                    
                    tracking_data[hand_key] = hand_data
        
        except Exception as e:
            print(f"Error getting Leap Motion data: {e}")
        
        return tracking_data
    
    def get_optitrack_data(self):
        """
        Get hand tracking data from OptiTrack
        
        Returns:
            Dictionary containing hand tracking data
        """
        tracking_data = {
            "left_hand": {"present": False},
            "right_hand": {"present": False}
        }
        
        try:
            frame = self.optitrack_client.wait_for_frame()
            
            if frame:
                # Process rigid bodies
                for rigid_body in frame.rigid_bodies:
                    # Map OptiTrack IDs to hand parts
                    # This mapping would need to be customized based on your OptiTrack setup
                    hand_mapping = {
                        1: {"hand": "left_hand", "part": "palm"},
                        2: {"hand": "right_hand", "part": "palm"},
                        # Add mappings for fingers
                        3: {"hand": "left_hand", "part": "thumb_tip"},
                        4: {"hand": "left_hand", "part": "index_tip"},
                        5: {"hand": "left_hand", "part": "middle_tip"},
                        6: {"hand": "left_hand", "part": "ring_tip"},
                        7: {"hand": "left_hand", "part": "pinky_tip"},
                        8: {"hand": "right_hand", "part": "thumb_tip"},
                        9: {"hand": "right_hand", "part": "index_tip"},
                        10: {"hand": "right_hand", "part": "middle_tip"},
                        11: {"hand": "right_hand", "part": "ring_tip"},
                        12: {"hand": "right_hand", "part": "pinky_tip"}
                    }
                    
                    if rigid_body.id in hand_mapping:
                        mapping = hand_mapping[rigid_body.id]
                        hand_key = mapping["hand"]
                        part = mapping["part"]
                        
                        # Mark hand as present
                        tracking_data[hand_key]["present"] = True
                        
                        # Process based on part type
                        if part == "palm":
                            # Get hand data
                            hand_data = tracking_data[hand_key]
                            
                            # Extract position and rotation
                            pos = rigid_body.position
                            rot = rigid_body.orientation
                            
                            # Filter palm position
                            filtered_pos = self.update_palm_position(
                                "left" if "left" in hand_key else "right",
                                pos,
                                rot
                            )
                            
                            # Store palm data
                            hand_data["palm"] = {
                                "position": filtered_pos.tolist(),
                                "rotation": rot.tolist() if rot is not None else [0, 0, 0, 1]
                            }
                            
                            # Initialize fingers if not present
                            if "fingers" not in hand_data:
                                hand_data["fingers"] = {}
                        
                        elif "_tip" in part:
                            # Extract finger name from part
                            finger_name = part.split("_")[0]
                            
                            # Get hand data
                            hand_data = tracking_data[hand_key]
                            
                            # Initialize fingers if not present
                            if "fingers" not in hand_data:
                                hand_data["fingers"] = {}
                            
                            # Initialize finger if not present
                            if finger_name not in hand_data["fingers"]:
                                hand_data["fingers"][finger_name] = {}
                            
                            # Extract position
                            pos = rigid_body.position
                            
                            # Filter finger position
                            filtered_pos = self.update_finger_position(
                                "left" if "left" in hand_key else "right",
                                finger_name,
                                "tip",
                                pos
                            )
                            
                            # Store finger tip data
                            hand_data["fingers"][finger_name]["tip"] = {
                                "position": filtered_pos.tolist()
                            }
        
        except Exception as e:
            print(f"Error getting OptiTrack data: {e}")
        
        return tracking_data
    
    def get_steamvr_data(self):
        """
        Get hand tracking data from SteamVR/OpenVR
        
        Returns:
            Dictionary containing hand tracking data
        """
        tracking_data = {
            "left_hand": {"present": False},
            "right_hand": {"present": False}
        }
        
        try:
            # SteamVR doesn't provide detailed finger tracking by default
            # This implementation focuses on controller positions
            
            # Get controller data
            left_controller = self.vr.devices.get("controller_1")
            right_controller = self.vr.devices.get("controller_2")
            
            if left_controller:
                pos, rot = left_controller.get_pose_euler()
                filtered_pos = self.update_palm_position("left", pos, rot)
                
                tracking_data["left_hand"] = {
                    "present": True,
                    "palm": {
                        "position": filtered_pos.tolist(),
                        "rotation": rot
                    },
                    # Add basic finger positions based on controller
                    "fingers": {
                        "index": {
                            "tip": {
                                "position": [pos[0], pos[1] - 0.03, pos[2] + 0.05]
                            }
                        },
                        "thumb": {
                            "tip": {
                                "position": [pos[0] - 0.03, pos[1], pos[2] + 0.03]
                            }
                        }
                    }
                }
            
            if right_controller:
                pos, rot = right_controller.get_pose_euler()
                filtered_pos = self.update_palm_position("right", pos, rot)
                
                tracking_data["right_hand"] = {
                    "present": True,
                    "palm": {
                        "position": filtered_pos.tolist(),
                        "rotation": rot
                    },
                    # Add basic finger positions based on controller
                    "fingers": {
                        "index": {
                            "tip": {
                                "position": [pos[0], pos[1] - 0.03, pos[2] + 0.05]
                            }
                        },
                        "thumb": {
                            "tip": {
                                "position": [pos[0] + 0.03, pos[1], pos[2] + 0.03]
                            }
                        }
                    }
                }
        
        except Exception as e:
            print(f"Error getting SteamVR data: {e}")
        
        return tracking_data
        
    def get_mediapipe_data(self, image):
        """
        Get hand tracking data from MediaPipe
        
        Args:
            image: Camera image
        
        Returns:
            Dictionary containing hand tracking data
        """
        tracking_data = {
            "left_hand": {"present": False},
            "right_hand": {"present": False}
        }
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # Process hand tracking
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # MediaPipe hand landmarks
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Determine hand side
                if results.multi_handedness:
                    hand_side = results.multi_handedness[idx].classification[0].label
                    is_left = (hand_side == "Left")
                else:
                    # Default to right hand if no classification
                    is_left = False
                
                hand_key = "left_hand" if is_left else "right_hand"
                tracking_data[hand_key]["present"] = True
                
                # Get palm position
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                middle_cmc = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                
                # Calculate palm position (using image dimensions to scale)
                palm_pos = [
                    (wrist.x + middle_cmc.x) / 2 * w,
                    (wrist.y + middle_cmc.y) / 2 * h,
                    (wrist.z + middle_cmc.z) / 2
                ]
                
                # Store palm data (no conversion needed)
                tracking_data[hand_key]["palm"] = {
                    "position": palm_pos,
                    "rotation": [0, 0, 0, 1]  # Default quaternion
                }
                
                # Initialize fingers
                tracking_data[hand_key]["fingers"] = {}
                
                # Process fingers
                finger_mapping = {
                    "thumb": [
                        self.mp_hands.HandLandmark.THUMB_CMC,
                        self.mp_hands.HandLandmark.THUMB_MCP,
                        self.mp_hands.HandLandmark.THUMB_IP,
                        self.mp_hands.HandLandmark.THUMB_TIP
                    ],
                    "index": [
                        self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                        self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
                        self.mp_hands.HandLandmark.INDEX_FINGER_DIP,
                        self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                    ],
                    "middle": [
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
                        self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP
                    ],
                    "ring": [
                        self.mp_hands.HandLandmark.RING_FINGER_MCP,
                        self.mp_hands.HandLandmark.RING_FINGER_PIP,
                        self.mp_hands.HandLandmark.RING_FINGER_DIP,
                        self.mp_hands.HandLandmark.RING_FINGER_TIP
                    ],
                    "pinky": [
                        self.mp_hands.HandLandmark.PINKY_MCP,
                        self.mp_hands.HandLandmark.PINKY_PIP,
                        self.mp_hands.HandLandmark.PINKY_DIP,
                        self.mp_hands.HandLandmark.PINKY_TIP
                    ]
                }
                
                joint_names = ["metacarpal", "proximal", "intermediate", "distal", "tip"]
                
                for finger_name, landmarks in finger_mapping.items():
                    finger_data = {}
                    
                    # For thumb, we need to handle the different joint structure
                    if finger_name == "thumb":
                        joint_names = ["metacarpal", "proximal", "distal", "tip"]
                    
                    for i, landmark_idx in enumerate(landmarks):
                        if i >= len(joint_names):
                            continue
                            
                        landmark = hand_landmarks.landmark[landmark_idx]
                        
                        # Store pixel coordinates directly (easier for visualization)
                        pos = [
                            landmark.x * w,
                            landmark.y * h,
                            landmark.z
                        ]
                        
                        finger_data[joint_names[i]] = {
                            "position": pos
                        }
                    
                    tracking_data[hand_key]["fingers"][finger_name] = finger_data
        
        return tracking_data
    
    def send_to_unity(self, data):
        """
        Send tracking data to Unity
        
        Args:
            data: Tracking data dictionary
        """
        if not self.unity_integration or self.unity_socket is None:
            return
        
        try:
            # Convert to JSON
            json_data = json.dumps(data)
            
            # Send data to Unity
            self.unity_socket.sendto(json_data.encode(), ("127.0.0.1", self.unity_port))
        except Exception as e:
            print(f"Error sending data to Unity: {e}")
    
    def run(self):
        """Main run loop for the hand tracking system"""
        self.running = True
        
        # Start a separate thread for Unity communication if enabled
        if self.unity_integration:
            unity_thread = threading.Thread(target=self.unity_communication_loop)
            unity_thread.daemon = True
            unity_thread.start()
        
        # For non-VR debug mode or fallback, use camera
        if not self.vr_initialized:
            self.run_camera_mode()
            return
        
        print(f"Running hand tracking with {self.tracking_system}")
        
        try:
            tracking_data = None
            
            # Main tracking loop
            while self.running:
                # Get hand tracking data from selected system
                if self.tracking_system == "leap":
                    tracking_data = self.get_leap_motion_data()
                elif self.tracking_system == "optitrack":
                    tracking_data = self.get_optitrack_data()
                elif self.tracking_system == "steamvr":
                    tracking_data = self.get_steamvr_data()
                
                # Print status
                left_present = tracking_data["left_hand"]["present"]
                right_present = tracking_data["right_hand"]["present"]
                print(f"Left hand: {'Detected' if left_present else 'Not detected'}, Right hand: {'Detected' if right_present else 'Not detected'}", end="\r")
                
                # Short sleep to prevent CPU hogging
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\nHand tracking stopped by user")
        except Exception as e:
            print(f"\nError in tracking loop: {e}")
        finally:
            self.running = False
            self.cleanup()
    
    def unity_communication_loop(self):
        """Separate thread for Unity communication"""
        while self.running:
            if self.tracking_system == "leap":
                tracking_data = self.get_leap_motion_data()
            elif self.tracking_system == "optitrack":
                tracking_data = self.get_optitrack_data()
            elif self.tracking_system == "steamvr":
                tracking_data = self.get_steamvr_data()
            else:
                tracking_data = None
            
            if tracking_data is not None:
                self.send_to_unity(tracking_data)
            
            time.sleep(0.01)  # Send at 100Hz
    
    def run_camera_mode(self):
        """Run in camera-based fallback mode using MediaPipe"""
        cap = cv2.VideoCapture(0)
        
        try:
            while self.running:
                success, img = cap.read()
                if not success:
                    continue
                
                # Flip image horizontally for selfie-view
                img = cv2.flip(img, 1)
                
                # Create a copy for visualization
                display_img = img.copy()
                
                # Process hand tracking directly with MediaPipe
                # Convert BGR to RGB for MediaPipe
                results = self.hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
                # Basic visualization from MediaPipe
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw native MediaPipe visualization
                        self.mp_draw.draw_landmarks(
                            display_img,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                            self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                        )
                
                # Process for our tracking data structure
                tracking_data = self.get_mediapipe_data(img)
                
                # Send data to Unity if enabled
                if self.unity_integration:
                    self.send_to_unity(tracking_data)
                
                # Additional custom visualization
                self.draw_debug_visualization(display_img, tracking_data)
                
                # Add status text
                left_present = tracking_data["left_hand"]["present"]
                right_present = tracking_data["right_hand"]["present"]
                status_text = f"Left hand: {'Detected' if left_present else 'Not detected'}, Right hand: {'Detected' if right_present else 'Not detected'}"
                cv2.putText(display_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
                
                # Display
                cv2.imshow("Hand Tracking (Camera Mode)", display_img)
                
                # Break loop with 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Hand tracking stopped by user")
        except Exception as e:
            print(f"Error in camera mode: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.cleanup()
                
    def draw_debug_visualization(self, img, tracking_data):
        """
        Draw debug visualization of hand tracking
        
        Args:
            img: Camera image
            tracking_data: Hand tracking data
        """
        for hand_key in ["left_hand", "right_hand"]:
            hand_data = tracking_data[hand_key]
            
            if not hand_data["present"]:
                continue
            
            # Draw palm
            if "palm" in hand_data:
                palm_pos = hand_data["palm"]["position"]
                palm_x = int(palm_pos[0])
                palm_y = int(palm_pos[1])
                cv2.circle(img, (palm_x, palm_y), 15, (0, 255, 0), 2)
                cv2.putText(img, f"{hand_key.split('_')[0].upper()} PALM", (palm_x - 20, palm_y - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw fingers
            if "fingers" in hand_data:
                for finger_name, finger_data in hand_data["fingers"].items():
                    # Color mapping for each finger
                    color = {
                        "thumb": (255, 0, 0),      # Red
                        "index": (0, 255, 0),      # Green
                        "middle": (0, 0, 255),     # Blue
                        "ring": (255, 255, 0),     # Yellow
                        "pinky": (0, 255, 255)     # Cyan
                    }.get(finger_name, (255, 255, 255))
                    
                    # Get joint positions
                    joint_positions = []
                    for joint_name in ["metacarpal", "proximal", "intermediate", "distal", "tip"]:
                        if joint_name in finger_data:
                            pos = finger_data[joint_name]["position"]
                            joint_positions.append((int(pos[0]), int(pos[1])))
                    
                    # Draw finger joints and connections
                    for i, pos in enumerate(joint_positions):
                        # Draw joint
                        cv2.circle(img, pos, 8, color, 1)
                        
                        # Draw connection to previous joint
                        if i > 0:
                            cv2.line(img, joint_positions[i-1], pos, color, 1)
                    
                    # Label finger tips
                    if joint_positions and finger_name:
                        tip_pos = joint_positions[-1]
                        cv2.putText(img, finger_name, (tip_pos[0] - 10, tip_pos[1] - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def cleanup(self):
        """Clean up resources"""
        if self.tracking_system == "leap" and hasattr(self, 'leap'):
            try:
                self.leap.close()
                print("Leap Motion connection closed")
            except:
                pass
        
        if self.unity_integration and self.unity_socket:
            try:
                self.unity_socket.close()
                print("Unity socket connection closed")
            except:
                pass

def main():
    """Main function to run the hand tracking system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hand Tracking System for VR")
    parser.add_argument("--system", type=str, default="leap", choices=["leap", "optitrack", "steamvr"], help="Tracking system to use")
    parser.add_argument("--unity", action="store_true", help="Enable Unity integration")
    parser.add_argument("--port", type=int, default=12345, help="Port for Unity communication")
    
    args = parser.parse_args()
    
    # Initialize hand tracking system
    tracking_system = HandTrackingSystem(
        tracking_system=args.system,
        unity_integration=args.unity,
        unity_port=args.port
    )
    
    # Run the tracking system
    tracking_system.run()

if __name__ == "__main__":
    main()