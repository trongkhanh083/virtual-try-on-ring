import argparse
import cv2
import numpy as np
from PIL import Image
import os

from hand_detector import HandDetector

class MaskHandTarget:
    def __init__(self):
        self.detector = HandDetector()

    def prepare_image(self, target_hand_path):
        """Load target hand as OpenCV + PIL"""
        cv_img = cv2.imread(target_hand_path)
        if cv_img is None:
            raise FileNotFoundError(f"Target hand image not found: {target_hand_path}")
        pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        target_filename = os.path.splitext(os.path.basename(target_hand_path))[0]
        return cv_img, pil_img, target_filename

    def estimate_ring_position(self, cv_img):
        """Use MediaPipe hand detector to estimate ring finger position with enhanced accuracy"""
        detections = self.detector.detect_hands(cv_img)
        if not detections:
            raise RuntimeError("No hand detected in target image")

        # Use first detected hand
        ring_info = self.detector.estimate_ring_position(detections[0], target_finger="ring")
        if ring_info.get("error"):
            raise RuntimeError(f"Error estimating ring position: {ring_info['error']}")

        return ring_info

    def create_ring_oval_mask(self, image_shape, ring_info, margin_factor=1.2):
        """
        Create an oval mask specifically for the ring position
        
        Args:
            image_shape: Shape of the target image
            ring_info: Dictionary containing ring position and orientation
            margin_factor: Factor to adjust the oval size
        """
        h, w = image_shape[:2]
        mask_array = np.zeros((h, w), dtype=np.uint8)

        mcp = ring_info["mcp_joint"]
        pip = ring_info["pip_joint"]
        x = int(0.4 * mcp[0] + 0.6 * pip[0])
        y = int(0.4 * mcp[1] + 0.6 * pip[1])
        finger_len = ring_info["finger_length"]
        
        # Calculate the angle perpendicular to the finger direction
        direction_angle = ring_info["direction_angle"]
        horizontal_angle = direction_angle + np.pi/2  # Rotate 90 degrees to be perpendicular to finger
        angle_deg = np.degrees(horizontal_angle)

        # Enhanced oval sizing based on finger anatomy
        # Oval dimensions proportional to finger length with adjustable margins
        ell_w = int(finger_len * 0.7 * margin_factor)  # Width along finger
        ell_h = int(finger_len * 0.4 * margin_factor)  # Height across finger

        # Draw the oval mask
        cv2.ellipse(mask_array, 
                   (int(x), int(y)), 
                   (ell_w // 2, ell_h // 2),
                   angle_deg, 0, 360, 255, -1)  # Filled white oval

        return Image.fromarray(mask_array)

    def create_debug_visualization(self, cv_img, ring_info, margin_factor=1.2):
        """
        Create a debug visualization showing hand landmarks and ring oval
        """
        debug_img = cv_img.copy()
        h, w = debug_img.shape[:2]
        
        # Draw hand landmarks if available
        detections = self.detector.detect_hands(cv_img)
        if detections and "landmarks" in detections[0]:
            landmarks = detections[0]["landmarks"]
            
            # Draw all landmarks
            for i, (x, y) in enumerate(landmarks):
                cv2.circle(debug_img, (x, y), 3, (0, 255, 0), -1)  # Green dots
                if i in [13, 14]:  # Ring finger landmarks
                    cv2.circle(debug_img, (x, y), 5, (0, 0, 255), -1)  # Red for ring finger
        
        # Draw ring oval with horizontal orientation
        mcp = ring_info["mcp_joint"]
        pip = ring_info["pip_joint"]
        x = int(0.4 * mcp[0] + 0.6 * pip[0])
        y = int(0.4 * mcp[1] + 0.6 * pip[1])
        finger_len = ring_info["finger_length"]
        
        # Calculate horizontal angle
        direction_angle = ring_info["direction_angle"]
        horizontal_angle = direction_angle + np.pi/2  # Rotate 90 degrees
        angle_deg = np.degrees(horizontal_angle)
        
        ell_w = int(finger_len * 0.7 * margin_factor)
        ell_h = int(finger_len * 0.4 * margin_factor)
        
        # Draw oval outline
        cv2.ellipse(debug_img, 
                   (int(x), int(y)), 
                   (ell_w // 2, ell_h // 2),
                   angle_deg, 0, 360, (255, 0, 0), 2)  # Blue outline
        
        # Draw center point
        cv2.circle(debug_img, (int(x), int(y)), 6, (0, 255, 255), -1)  # Yellow center
        
        # Draw direction lines
        mcp = ring_info["mcp_joint"]
        pip = ring_info["pip_joint"]
        
        # Original finger direction
        cv2.line(debug_img, mcp, pip, (255, 255, 0), 2)
        
        # Add text showing the angles
        cv2.putText(debug_img, f"Finger angle: {np.degrees(direction_angle):.1f}°", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(debug_img, f"Oval angle: {angle_deg:.1f}°", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return Image.fromarray(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))

def main():
    parser = argparse.ArgumentParser(description="Generate ring oval mask for target hand using MediaPipe landmarks")
    parser.add_argument('--target', '-t', type=str, required=True, help="Path to target hand image")
    parser.add_argument('--output', '-o', type=str, required=True, help="Output folder")
    parser.add_argument('--margin', '-m', type=float, default=0.8, help="Margin factor for oval size")
    args = parser.parse_args()

    masker = MaskHandTarget()

    # Load image
    cv_img, pil_img, target_filename = masker.prepare_image(args.target)

    # Estimate ring placement
    ring_info = masker.estimate_ring_position(cv_img)
    print(f"Ring info: {ring_info}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Basic ring oval mask
    ring_mask = masker.create_ring_oval_mask(cv_img.shape, ring_info, args.margin)
    ring_mask_path = f"{args.output}/{target_filename}_mask.png"
    ring_mask.save(ring_mask_path)
    print(f"Saved ring oval mask: {ring_mask_path}")

    # Debug visualization
    debug_viz = masker.create_debug_visualization(cv_img, ring_info, args.margin)
    debug_path = f"{args.output}/{target_filename}_debug.png"
    debug_viz.save(debug_path)
    print(f"Saved debug visualization: {debug_path}")

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)