import gradio as gr
import os
from PIL import Image
import numpy as np
import cv2

from ring_segmenter import process_image as segment_ring
from mask_hand_target import MaskHandTarget
from ring_compositing import traditional_compositing
from virtual_try_on import main as refine_pipeline

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Paths to bare hand templates
LEFT_HAND_DEFAULT = "target/left.jpg"
RIGHT_HAND_DEFAULT = "target/right.jpg"

# Setup Real-ESRGAN
def setup_realesrgan():
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path="Real-ESRGAN/weights/RealESRGAN_x4plus.pth",
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        gpu_id=None
    )
    return upsampler

upsampler = setup_realesrgan()

def upscale_image(pil_img):
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    output, _ = upsampler.enhance(img, outscale=3.5)
    return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

def tryon_pipeline(hand_with_ring_img, bare_hand_img):
    os.makedirs("workspace", exist_ok=True)

    # Save uploaded
    uploaded_path = "workspace/uploaded.jpg"
    hand_with_ring_img.save(uploaded_path)

    # Step 1: segment ring
    mask_path = "workspace/ring_mask.png"
    ring_path = "workspace/ring_isolated.png"
    segment_ring(uploaded_path, output_folder="workspace", mask_folder="workspace", ring_folder="workspace")
    print("Step 1: Ring segmented.")

    # Step 2: generate oval masks for bare hands
    if bare_hand_img is not None:
        bare_hands = [("custom", bare_hand_img)]
    else:
        bare_hands = [("left", LEFT_HAND_DEFAULT), ("right", RIGHT_HAND_DEFAULT)]

    masker = MaskHandTarget()
    results = {}
    for side, bare_hand in bare_hands:
        cv_img, _, _ = masker.prepare_image(bare_hand)
        ring_info = masker.estimate_ring_position(cv_img)
        oval_mask = masker.create_ring_oval_mask(cv_img.shape, ring_info)
        mask_save = f"workspace/{side}_mask.png"
        oval_mask.save(mask_save)
        print("Step 2: Bare hand mask created.")

        # Step 3: composite isolated ring
        composite = traditional_compositing(bare_hand, ring_path, mask_save)
        composite_path = f"workspace/{side}_composite.jpg"
        composite.save(composite_path)
        print("Step 3: Ring composited to bare hand.")

        # Step 4: refine with SD
        refined = refine_pipeline(bare_hand, mask_save, composite_path)
        refined_path = f"workspace/{side}_refined.jpg"
        refined.save(refined_path)
        print("Step 4: Image refined with SD1.5 + Controlnet.")

        # Step 5: upscale with Real-ESRGAN
        final_img = upscale_image(refined)
        final_path = f"workspace/{side}_final.jpg"
        final_img.save(final_path)
        results[side] = final_img
        print("Step 5: Image upscaled with Real-ESRGAN.")

    return results["left"], results["right"] or results["custom"]

def validated_input(hand_with_ring, bare_hand):
    if hand_with_ring is None:
        raise gr.Error("❌ Please upload a hand wearing a ring before running.")
    if bare_hand is None:
        raise gr.Error("❌ Please upload a bare hand before running.")
    return tryon_pipeline(hand_with_ring, bare_hand)

# Gradio Interface
with gr.Blocks(css=".center-text {text-align: center;}") as demo:
    gr.Markdown("# 💍 Virtual Try-On Ring with ControlNet Refinement", elem_classes="center-text")

    with gr.Row():
        # Left column
        with gr.Column(scale=1):
            hand_with_ring = gr.Image(type="pil", label="Upload Hand With Ring")
            bare_hand = gr.Image(type="pil", label="Upload Your Bare Hand")

            with gr.Row():
                clear_btn = gr.Button("Clear")
                run_btn = gr.Button("Run", variant="primary")

            # Example group 1: Hands with rings
            gr.Examples(
                examples=[
                    ["assets/2.jpg", None],
                    ["assets/4.jpg", None],
                    ["assets/5.jpg", None],
                    ["assets/10.jpg", None],
                    ["assets/12.jpg", None],
                    ["assets/14.jpg", None],
                    ["assets/15.jpg", None],
                    ["assets/19.jpg", None],
                ],
                inputs=[hand_with_ring, bare_hand],
                label="📸 Example Ring Hands"
            )

            # Example group 2: Bare hands
            gr.Examples(
                examples=[
                    [None, "target/left.jpg"],
                    [None, "target/right.jpg"],
                ],
                inputs=[hand_with_ring, bare_hand],
                label="✋ Example Bare Hands"
            )

        # Right column
        with gr.Column(scale=1):
            output = gr.Image(type="pil", label="Virtual Try-On Result")

    # Button logic
    run_btn.click(
        fn=validated_input,
        inputs=[hand_with_ring, bare_hand],
        outputs=output
    )
    clear_btn.click(
        lambda: (None, None, None),
        inputs=None,
        outputs=[hand_with_ring, bare_hand, output],
        queue=False
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
    )

