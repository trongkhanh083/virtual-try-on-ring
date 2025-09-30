import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Dict
import os
import argparse

class HandDetector:
    def __init__(self):
        """
        Initialize the Hand Detector for hand detection and ring position estimation
        """
        # Initialize MediaPipe Hands for hand detection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Finger landmark indices
        self.FINGER_INDICES = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        print("Hand Detector initialized")

    def detect_hands(self, image: np.ndarray) -> List[Dict]:
        """
        Detect hands and extract landmarks using MediaPipe
        
        Returns:
            List of hand detections with landmarks and bounding boxes
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hand_detections = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w = image.shape[:2]
                
                # Extract all landmark coordinates
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    landmarks.append((x, y))
                
                # Calculate bounding box
                x_coords = [lm[0] for lm in landmarks]
                y_coords = [lm[1] for lm in landmarks]
                bbox = {
                    'x_min': max(0, min(x_coords) - 20),
                    'y_min': max(0, min(y_coords) - 20),
                    'x_max': min(w, max(x_coords) + 20),
                    'y_max': min(h, max(y_coords) + 20)
                }
                
                hand_detections.append({
                    'landmarks': landmarks,
                    'bbox': bbox,
                    'confidence': 0.9  # Default confidence since we're not using handedness
                })
        
        return hand_detections

    def estimate_ring_position(self, hand_detection: Dict, 
                         target_finger: str = 'ring') -> Dict:
        """
        Safely estimate ring position with proper error handling
        """
        landmarks = hand_detection['landmarks']
        
        # Check if we have enough landmarks
        if len(landmarks) < 21:
            return {
                'position': (0, 0),
                'finger': target_finger,
                'confidence': 0.0,
                'error': f'Insufficient landmarks: {len(landmarks)}/21'
            }
        
        try:
            # MediaPipe ring finger landmarks:
            # 13: MCP (Metacarpophalangeal joint)
            # 14: PIP (Proximal Interphalangeal joint) 
            mcp_joint = landmarks[13]  # Base of ring finger
            pip_joint = landmarks[14]  # First knuckle
            
            # Calculate midpoint between MCP and PIP for ring position
            ring_x = int(0.5 * mcp_joint[0] + 0.5 * pip_joint[0])
            ring_y = int(0.5 * mcp_joint[1] + 0.5 * pip_joint[1])
            
            # Calculate direction vector
            dx = pip_joint[0] - mcp_joint[0]
            dy = pip_joint[1] - mcp_joint[1]
            direction_angle = np.arctan2(dy, dx)

            finger_length = np.linalg.norm(np.array(pip_joint) - np.array(mcp_joint))
            
            confidence = self.calculate_ring_confidence(hand_detection, [mcp_joint, pip_joint])
            
            return {
                'position': (ring_x, ring_y),
                'finger': target_finger,
                'confidence': confidence,
                'mcp_joint': mcp_joint,
                'pip_joint': pip_joint,
                'direction_angle': direction_angle,
                'finger_length': finger_length,
                'landmark_indices': [13, 14],
                'error': None
            }
            
        except IndexError as e:
            return {
                'position': (0, 0),
                'finger': target_finger,
                'confidence': 0.0,
                'error': f'Landmark index error: {str(e)}'
            }

    def calculate_ring_confidence(self, hand_detection: Dict, 
                            finger_landmarks: List[Tuple]) -> float:
        """Calculate confidence score for ring position estimation"""
        landmarks = hand_detection['landmarks']
        
        # Check if finger landmarks are within image bounds
        h, w = 480, 640  # Assuming standard size, adjust if needed
        in_bounds = all(0 <= x < w and 0 <= y < h for x, y in finger_landmarks)
        
        # Use the actual landmarks passed instead of assuming [0, 3] indices
        if len(finger_landmarks) >= 2:
            base = finger_landmarks[0]
            tip = finger_landmarks[-1]  # Use last landmark instead of fixed index 3
            verticality = abs(tip[1] - base[1]) / (abs(tip[0] - base[0]) + 1e-6)
            
            # Combine factors
            confidence = 0.7 if in_bounds else 0.3
            confidence *= min(verticality / 2.0, 1.0)  # Normalize verticality
        else:
            confidence = 0.3  # Low confidence if not enough landmarks
        
        return min(confidence, 1.0)


def generate_output_path(input_path: str, output_folder: str):
    """
    Generate automatic filename based on input filename
    """
    # Get input filename without extension
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create output path
    output_path = os.path.join(output_folder, f"{input_filename}.jpg")
    
    return output_path


def process_image(image_path: str, output_folder: str = None):
    """
    Process a single image for hand detection and ring position estimation
        
    Args:
        image_path: Path to input image
        output_folder: Output folder for result images
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found")
        return False
        
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return False
        
    print(f"Processing image: {image_path}")
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Create output folder if it doesn't exist
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        output_path = generate_output_path(image_path, output_folder)
    else:
        output_path = None
        
    # Initialize detector
    detector = HandDetector()
        
    # Detect hands
    hand_detections = detector.detect_hands(image)
        
    print(f"Hand detected: {len(hand_detections)}")
        
    # Create a copy for drawing results
    result_image = image.copy()
        
    for i, hand in enumerate(hand_detections):
        print(f"Confidence: {hand['confidence']:.2f}")
            
        # Draw bounding box
        bbox = hand['bbox']
        cv2.rectangle(result_image,
                    (bbox['x_min'], bbox['y_min']), 
                    (bbox['x_max'], bbox['y_max']), 
                    (0, 255, 0), 2)
            
        # Draw all landmarks
        for landmark in hand['landmarks']:
            cv2.circle(result_image, landmark, 3, (255, 0, 0), -1)
            
        # Estimate ring position
        ring_info = detector.estimate_ring_position(hand)
            
        if ring_info['error'] is None:
            # Draw ring position
            ring_pos = ring_info['position']
            cv2.circle(result_image, ring_pos, 8, (0, 0, 255), -1)
                
            # Draw line showing ring direction
            mcp = ring_info['mcp_joint']
            pip = ring_info['pip_joint']
            cv2.line(result_image, mcp, pip, (0, 255, 255), 2)
                
            print(f"Ring position: {ring_pos}")
            print(f"Ring confidence: {ring_info['confidence']:.2f}")
            print(f"Direction angle: {ring_info['direction_angle']:.2f} radians")
        else:
            print(f"Ring estimation error: {ring_info['error']}")
            
        # Display information on image
        cv2.putText(result_image, f"Hand {i+1}", 
                (bbox['x_min'], bbox['y_min'] - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
        if ring_info['error'] is None:
            cv2.putText(result_image, f"Ring: {ring_info['confidence']:.2f}", 
                    (bbox['x_min'], bbox['y_min'] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
    # Save result
    if output_path:
        cv2.imwrite(output_path, result_image)
        print(f"Output image saved to: {output_path}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand Detection and Ring Position Estimation")
    parser.add_argument('--input', '-i', required=True, help="Input image path")
    parser.add_argument('--output', '-o', required=True, help="Output folder")

    args = parser.parse_args()

    success = process_image(args.input, args.output)
    exit(0 if success else 1)