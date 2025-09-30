import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from typing import Tuple, Dict
import urllib.request
import os
import argparse

from hand_detector import HandDetector

class RingSegmenter:
    def __init__(self, sam_model_type: str = "vit_b", sam_checkpoint_path: str = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the Ring Segmenter with SAM
        
        Args:
            sam_model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
            sam_checkpoint_path: Path to SAM checkpoint file
            device: Device to run models on ('cuda' or 'cpu')
        """
        self.device = device
        self.sam_model_type = sam_model_type
        
        # Initialize SAM
        if sam_checkpoint_path is None:
            # Download checkpoint if not provided
            sam_checkpoint_path = self.download_sam_checkpoint()
            
        self.sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        
        print(f"Ring Segmenter initialized with {sam_model_type} on {device}")

    def download_sam_checkpoint(self) -> str:
        """Download SAM checkpoint if not available locally"""
        
        checkpoints = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        checkpoint_path = f"sam_{self.sam_model_type}.pth"
        
        if not os.path.exists(checkpoint_path):
            print(f"Downloading SAM {self.sam_model_type} checkpoint...")
            urllib.request.urlretrieve(checkpoints[self.sam_model_type], checkpoint_path)
            print("Download completed!")
        
        return checkpoint_path

    def create_ring_bounding_box(self, ring_position: Tuple[int, int], 
                           direction_angle: float,
                           image_shape: Tuple[int, int],
                           box_size: int = 80) -> np.ndarray:
        """
        Create an oriented bounding box aligned with finger direction
        """
        x, y = ring_position
        h, w = image_shape[:2]
        
        # Create a more accurate box aligned with finger direction
        cos_angle = np.cos(direction_angle)
        sin_angle = np.sin(direction_angle)
        
        # Box dimensions (longer along finger direction)
        length = box_size * 1.0  # Longer along finger
        width = box_size * 1.0   # Narrower across finger
        
        # Calculate corner points of rotated rectangle
        corners = []
        for dx, dy in [(-length/2, -width/2), (-length/2, width/2),
                    (length/2, width/2), (length/2, -width/2)]:
            # Rotate the point
            rotated_dx = dx * cos_angle - dy * sin_angle
            rotated_dy = dx * sin_angle + dy * cos_angle
            corners.append((x + rotated_dx, y + rotated_dy))
        
        # Get axis-aligned bounding box of rotated rectangle
        x_coords = [corner[0] for corner in corners]
        y_coords = [corner[1] for corner in corners]
        
        x_min = max(0, int(min(x_coords)))
        y_min = max(0, int(min(y_coords)))
        x_max = min(w, int(max(x_coords)))
        y_max = min(h, int(max(y_coords)))
        
        return np.array([x_min, y_min, x_max, y_max])
    
    def segment_ring(self, image: np.ndarray, 
                    ring_position: Tuple[int, int],
                    direction_angle: float = 0.0,
                    finger_length: float = 40.0,
                    hand_bbox: Dict = None) -> Dict:
        """
        Segment ring using SAM with position and direction from hand detector
        """
        try:
            # self.predictor.set_image(resized_img)
            self.predictor.set_image(image)
                
            # Validate ring position
            h, w = image.shape[:2]
            x, y = ring_position
            
            if not (0 <= x < w and 0 <= y < h):
                # If position is out of bounds, use image center
                ring_position = (w // 2, h // 2)
                print(f"Warning: Ring position out of bounds, using center: {ring_position}")
            
            # Adaptive box size
            if hand_bbox is not None:
                hand_w = hand_bbox["x_max"] - hand_bbox["x_min"]
                hand_h = hand_bbox["y_max"] - hand_bbox["y_min"]
                hand_size = max(hand_w, hand_h)
                
                # Scale box relative to hand size
                box_size = int(0.18 * hand_size)
                print(f"Use hand size for calculate box: {box_size}")
            else:
                # fallback: use finger length
                box_size = int(finger_length * 1.8)
                print(f"Fallback finger length to calculate box {box_size}")

            # Adaptive oriented box
            # box_size = int(max(40, min(120, finger_length * 1.8)))
            box_size = int(max(40, min(120, box_size)))
            box_prompt = self.create_ring_bounding_box(
                ring_position, direction_angle, image.shape, box_size
            )

            # Get masks from SAM
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_prompt,
                multimask_output=True,
            )
            
            # Check if we got any masks
            if len(masks) == 0 or len(scores) == 0:
                raise ValueError("SAM returned no masks")
            
            # Choose best mask: highest score AND contains the ring position
            scaled_x, scaled_y = ring_position
            best_idx = -1
            best_score = -1
            for i, (mask, score) in enumerate(zip(masks, scores)):
                if mask[scaled_y, scaled_x] and score > best_score:
                    best_idx = i
                    best_score = score
            if best_idx == -1:  # fallback to highest score
                best_idx = int(np.argmax(scores))
                best_score = scores[best_idx]

            best_mask = masks[best_idx]
            
            # Post-process the mask
            processed_mask = self.post_process_mask(best_mask, image.shape)
                
            return {
                'masks': masks,
                'scores': scores,
                'best_mask': best_mask,
                'processed_mask': processed_mask,
                'best_score': best_score,
                'prompt_point': ring_position,
                'prompt_box': box_prompt,
                'error': None
            }
            
        except Exception as e:
            print(f"Segmentation error: {e}")
            # Return a default empty result
            empty_mask = np.zeros(image.shape[:2], dtype=bool)
            return {
                'masks': [empty_mask],
                'scores': [0.0],
                'best_mask': empty_mask,
                'processed_mask': empty_mask.astype(np.uint8) * 255,
                'best_score': 0.0,
                'prompt_point': ring_position,
                'prompt_box': None,
                'error': str(e)
            }

    def post_process_mask(self, mask: np.ndarray, image_shape,
                         min_area_ratio: int = 0.00005,
                         max_area_ratio: int = 0.2) -> np.ndarray:
        """
        Clean up the mask using morphological operations and area filtering
        """
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Morphological operations to clean up small noise
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours and filter by area
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask_cleaned
        
        h, w = image_shape[:2]
        min_area = int(min_area_ratio * h * w)
        max_area = int(max_area_ratio * h * w)
            
        # Create new mask from filtered contours
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                filtered_contours.append(contour)

        final_mask = np.zeros_like(mask_cleaned)
        if filtered_contours:
            cv2.drawContours(final_mask, filtered_contours, -1, 255, -1)
        
        return final_mask

    def draw_segmentation_result(self, image: np.ndarray, segmentation_result: Dict) -> np.ndarray:
        """
        Draw segmentation results on the image
        """
        result_image = image.copy()
        
        # Draw prompt box
        if segmentation_result['prompt_box'] is not None:
            x_min, y_min, x_max, y_max = segmentation_result['prompt_box']
            cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # Draw prompt point
        point = segmentation_result['prompt_point']
        cv2.circle(result_image, point, 8, (0, 0, 255), -1)
        cv2.circle(result_image, point, 10, (255, 255, 255), 2)
        
        # Draw segmentation mask
        mask = segmentation_result['processed_mask']
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green color for mask
        
        # Blend mask with image
        result_image = cv2.addWeighted(result_image, 0.7, colored_mask, 0.3, 0)
        
        # Draw contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)
        
        # Add text information
        score = segmentation_result['best_score']
        cv2.putText(result_image, f"SAM Score: {score:.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_image, f"SAM Score: {score:.3f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        
        return result_image

    def save_mask(self, mask: np.ndarray, output_path: str):
        """
        Save the mask as an image file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Ensure the mask is in the correct format for saving
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # If mask is boolean, convert to 0-255
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        
        success = cv2.imwrite(output_path, mask)
        if success:
            print(f"Mask saved to: {output_path}")
        else:
            print(f"Error: Failed to save mask to {output_path}")

    def extract_ring_region(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Extract the ring region from the original image using the mask
        """
        # Create a black background
        result = np.zeros_like(image)
        
        # Apply mask to extract ring region
        result[mask > 0] = image[mask > 0]
        
        return result


def generate_output_paths(input_path: str, output_folder: str, mask_folder: str = None, ring_folder: str = None):
    """
    Generate automatic filenames based on input filename
    """
    # Get input filename without extension
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    # Create output paths
    output_image_path = os.path.join(output_folder, f"{input_filename}.jpg")
    
    output_mask_path = None
    if mask_folder:
        output_mask_path = os.path.join(mask_folder, f"{input_filename}.png")
    
    output_ring_path = None
    if ring_folder:
        output_ring_path = os.path.join(ring_folder, f"{input_filename}.png")
    
    return output_image_path, output_mask_path, output_ring_path


def process_image(image_path: str, output_folder: str = None, 
                  mask_folder: str = None, ring_folder: str = None):
    """
    Complete pipeline: detect hand -> estimate ring position -> segment ring
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
    
    # Create folders if they don't exist
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    if mask_folder:
        os.makedirs(mask_folder, exist_ok=True)
    if ring_folder:
        os.makedirs(ring_folder, exist_ok=True)
    
    # Generate output paths
    output_image_path, output_mask_path, output_ring_path = generate_output_paths(
        image_path, output_folder, mask_folder, ring_folder
    )
    
    # Step 1: Detect hand and estimate ring position
    hand_detector = HandDetector()
    hand_detections = hand_detector.detect_hands(image)
    
    if len(hand_detections) == 0:
        print("No hands detected in the image")
        return False
    
    print(f"Hand detected: {len(hand_detections)}")
    
    # Step 2: Initialize ring segmenter
    ring_segmenter = RingSegmenter()
    
    # Process each detected hand
    result_images = []
    
    for i, hand in enumerate(hand_detections):
        
        # Estimate ring position
        ring_info = hand_detector.estimate_ring_position(hand)
        
        if ring_info['error'] is not None:
            print(f"Ring estimation error: {ring_info['error']}")
            continue
        
        ring_position = ring_info['position']
        direction_angle = ring_info['direction_angle']
        
        print(f"Ring position: {ring_position}")
        print(f"Direction angle: {direction_angle:.2f} radians")
        
        # Step 3: Segment the ring using SAM
        segmentation_result = ring_segmenter.segment_ring(
            image, ring_position, direction_angle,
            finger_length=ring_info['finger_length'],
            hand_bbox=hand['bbox']
        )
        
        if segmentation_result['error'] is not None:
            print(f"Segmentation error: {segmentation_result['error']}")
            continue
        
        print(f"SAM segmentation score: {segmentation_result['best_score']:.3f}")
        
        # Step 4: Draw results
        result_image = ring_segmenter.draw_segmentation_result(image, segmentation_result)
        result_images.append(result_image)
        
        # Save mask if requested
        if output_mask_path:
            # Create unique mask filename for each hand if multiple hands
            if len(hand_detections) > 1:
                base, ext = os.path.splitext(output_mask_path)
                hand_mask_path = f"{base}_hand{i+1}{ext}"
            else:
                hand_mask_path = output_mask_path
                
            ring_segmenter.save_mask(segmentation_result['processed_mask'], hand_mask_path)
        
        # Save isolated ring if requested
        if output_ring_path:
            # Create unique ring filename for each hand if multiple hands
            if len(hand_detections) > 1:
                base, ext = os.path.splitext(output_ring_path)
                hand_ring_path = f"{base}_hand{i+1}{ext}"
            else:
                hand_ring_path = output_ring_path
                
            ring_region = ring_segmenter.extract_ring_region(image, segmentation_result['processed_mask'])
            cv2.imwrite(hand_ring_path, ring_region)
            print(f"Isolated ring saved to: {hand_ring_path}")
    
    # Combine all results or use the first one
    if result_images:
        final_result = result_images[0]
        
        # Save final result image
        if output_image_path:
            cv2.imwrite(output_image_path, final_result)
            print(f"Result image saved to: {output_image_path}")
    
        return True
    else:
        print("No successful ring segmentations")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ring Segmentation with Hand Detection")
    parser.add_argument('--input', '-i', required=True, help="Input image path")
    parser.add_argument('--output', '-o', required=True, help="Output folder")
    parser.add_argument('--mask', '-m', help="Output folder for masks (optional)")
    parser.add_argument('--ring', '-r', help="Output folder for isolated ring (optional)")
    
    args = parser.parse_args()
    
    success = process_image(
        args.input, 
        args.output, 
        args.mask, 
        args.ring
    )
    exit(0 if success else 1)