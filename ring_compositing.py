import cv2
import numpy as np
from PIL import Image
from rembg import remove
import argparse
import os

def crop_to_content(img_rgba):
    """Crop transparent/black edges from an RGBA image"""
    gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgba
    x, y, w, h = cv2.boundingRect(contours[0])
    return img_rgba[y:y+h, x:x+w]

def normalize_ring_orientation(img_rgba):
    """Detect ring tilt angle and rotate to horizontal"""
    gray = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_rgba

    # Use largest contour (the ring)
    cnt = max(contours, key=cv2.contourArea)

    # Fit ellipse to contour to get orientation
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        angle = ellipse[2]  # angle of major axis
    else:
        angle = 0

    # Normalize angle: ring horizontal (0° or 180°)
    if angle > 90:
        angle = angle + 90
    else:
        angle = angle - 90

    # Rotate image
    (h, w) = img_rgba.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(img_rgba, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_TRANSPARENT)
    return rotated

def traditional_compositing(hand_path, ring_path, mask_path):
    """
    Place a ring onto the hand using a mask for positioning.
    """
    # Load images
    hand_img = Image.open(hand_path).convert("RGB")
    ring_img = Image.open(ring_path).convert("RGBA")
    mask_img = Image.open(mask_path).convert("L")

    ring_img_rembg = remove(ring_img)

    # Convert to numpy arrays
    hand_np = np.array(hand_img)
    ring_np = np.array(ring_img_rembg)
    mask_np = np.array(mask_img, dtype=np.uint8)

    # Ensure mask is binary (0 and 255)
    mask_binary = (mask_np > 128).astype(np.uint8) * 255

    # Find bounding box of the mask
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in mask")
    x, y, w, h = cv2.boundingRect(contours[0])

    # Crop + deskew ring to horizontal before resizing
    ring_np = crop_to_content(ring_np)
    ring_np = normalize_ring_orientation(ring_np)

    # Resize ring to match mask dimensions
    ring_resized = cv2.resize(ring_np, (w, h), interpolation=cv2.INTER_AREA)

    # Separate alpha channel if exists
    if ring_resized.shape[2] == 4:
        ring_alpha = ring_resized[:, :, 3] / 255.0
        ring_rgb = ring_resized[:, :, :3]
    else:
        ring_alpha = np.ones((h, w)) * 0.9
        ring_rgb = ring_resized

    # Alpha blending
    result_np = hand_np.copy()
    for c in range(3):
        result_np[y:y+h, x:x+w, c] = (
            ring_alpha * ring_rgb[:, :, c] +
            (1 - ring_alpha) * result_np[y:y+h, x:x+w, c]
        )

    return Image.fromarray(result_np)

def main():
    parser = argparse.ArgumentParser(description="Composite the ring to exact place on finger")
    parser.add_argument('--finger', '-f', type=str, required=True, help="Path to the bare hand")
    parser.add_argument('--mask', '-m', type=str, required=True, help="Path to the mask ring position in bare hand")
    parser.add_argument('--ring', '-r', type=str, required=True, help="Path to the isolated ring")
    parser.add_argument('--output', '-o', type=str, required=True, help="Path to output folder")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    output_filename = os.path.splitext(os.path.basename(args.ring))[0]

    composite_result = traditional_compositing(args.finger, args.ring, args.mask)
    composite_path = f"{args.output}/{output_filename}.jpg"
    composite_result.save(composite_path)
    print(f"Saved composite image: {composite_path}")

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)