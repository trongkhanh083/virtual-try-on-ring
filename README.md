💍 Virtual Ring Try-On

An AI-powered Virtual Ring Try-On system that lets users visualize how a ring looks on their hand.
The pipeline combines computer vision and generative AI techniques:

Mediapipe Hand Detection – detect and localize hand.

Segment Anything (SAM) – isolate the ring region.

Traditional Compositing – place the ring onto bare hand templates.

Stable Diffusion + ControlNet (Inpainting) – refine compositing with realistic shadows, reflections, and lighting.

Real-ESRGAN – upscale the final output for sharp, high-quality results.

✨ Features

Upload a hand wearing a ring → the system extracts the ring and transfers it to a bare hand.

Supports both left and right hand templates (default examples included).

Optionally, users can upload their own bare hand for try-on.

Outputs high-resolution, photo-realistic results thanks to ESRGAN enhancement.

Gradio-powered interactive web UI.

🖼️ Demo
<p align="center"> <img src="assets/demo_example.png" width="700"/> </p>

Input: hand with ring

Output: virtual try-on on left & right bare hands

🚀 Installation

Clone this repo:

git clone https://github.com/<your-username>/virtual-ring-tryon.git
cd virtual-ring-tryon


Create a conda or venv environment:

conda create -n ringvton python=3.10 -y
conda activate ringvton


Install dependencies:

pip install -r requirements.txt

⚙️ Usage
Run the Gradio app locally
python app.py


This launches a local web interface at http://127.0.0.1:7860/.

🖥️ Hardware Requirements

GPU strongly recommended

Tested on RTX 2070 8GB (works fine with optimizations).

At least 6GB VRAM for Stable Diffusion + ESRGAN.

CPU-only mode is not recommended (too slow, may OOM).
