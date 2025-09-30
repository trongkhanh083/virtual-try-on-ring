from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers import UniPCMultistepScheduler
import torch
import cv2
from PIL import Image
import numpy as np
import os
import argparse

def setup_refinement_pipeline():
    """
    Setup the AI pipeline for refinement only
    """
    # Use Canny Edge ControlNet to preserve hand structure
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_canny", 
        torch_dtype=torch.float16
    )
    
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "Sanster/Realistic_Vision_V1.4-inpainting",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    
    return pipe

def create_canny_control_image(image):
    """
    Create canny edge map to preserve hand structure
    """
    image_np = np.array(image)
    # Convert to grayscale if needed
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
        
    edges = cv2.Canny(gray, 50, 150)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges_rgb)

def ai_refinement(composite_img, mask_img, hand_img, ring_description=""):
    """
    Step 2: Use AI to naturally blend and refine the composite
    """
    pipe = setup_refinement_pipeline()
    
    # Create control image from original hand (to preserve structure)
    control_image = create_canny_control_image(hand_img)
    
    # Craft refinement-specific prompts
    prompt = """
    highly realistic jewelry try-on, preserve original ring design, seamless integration with skin, 
    natural soft shadows wrapping around finger, realistic metal reflection and highlights, 
    perfect lighting consistency with hand, ultra-photorealistic detail, professional product photography
    """

    negative_prompt = """
    altered ring design, different jewelry, extra rings, distorted shapes, warped ring, 
    finger deformation, skin artifacts, low resolution, blurry, painting, sketch, cartoon, CGI, 
    fake-looking, watermark, text
    """
    
    # Generate with LOW strength - we just want blending, not new content
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=composite_img,      # Our traditional composite
        mask_image=mask_img,      # Mask around the ring area
        control_image=control_image,
        strength=0.1,            # LOW strength - just refine blending
        controlnet_conditioning_scale=0.7,
        num_inference_steps=15,   # Fewer steps for subtle changes
        guidance_scale=6.0,       # Lower guidance for subtlety
        generator=torch.manual_seed(42)  # For consistency
    ).images[0]
    
    return result

def main(hand_path, mask_path, composite_path):
    """
    Complete hybrid pipeline
    """
    # Load images
    hand_img = Image.open(hand_path).convert("RGB")
    composite_img = Image.open(composite_path).convert("RGB")
    mask_img = Image.open(mask_path).convert("L")

    os.makedirs("result", exist_ok=True)
    
    print("AI refinement...")
    # control_image = create_canny_control_image(hand_img)
    # control_image.save("output/control_image.jpg")
    final_result = ai_refinement(composite_img, mask_img, hand_img)
    
    return final_result

# final_result = main(
#     hand_path="target/left.jpg",
#     mask_path="target/left_mask.png",
#     composite_path="composite/5.jpg"
# )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Virtual Try-on Ring")
    parser.add_argument('--source', '-s', type=str, required=True, default='1.jpg', help="Path to the bare hand image")
    parser.add_argument('--mask', '-m', type=str, required=True, default='1_ring.png', help="Path to the ring mask image")
    parser.add_argument('--composite', '-c', type=str, required=True, help="Path to the composite image")
    parser.add_argument('--output', '-o', type=str, required=True, help="Path to output folder")
    args = parser.parse_args()

    final_result = main(
        hand_path=args.source,
        mask_path=args.mask,
        composite_path=args.composite
    )
    output_filename = os.path.splitext(os.path.basename(args.ring))[0]
    result_path = f"{args.output}/{output_filename}.jpg"
    final_result.save(result_path)
    print(f"Saved final result to: {result_path}")

    success = final_result()
    exit(0 if success else 1)