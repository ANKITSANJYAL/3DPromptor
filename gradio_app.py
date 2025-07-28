import sys, os
sys.path.append(os.path.abspath("taming-transformers"))


import gradio as gr
import torch
from PIL import Image
import os
from app.inference import run_promptfusion3d
from app.utils import export_for_gaussian_splatting

DESCRIPTION = """
# 🧱 PromptFusion3D
Generate stylized 3D objects from:
- 📝 Prompt only: "cat in lego style"
- 🖼️ Image + Prompt: Upload image + "in clay style"

Uses SDXL + Local Prompt Adaptation + Zero123-XL + Gaussian Splatting.
"""

def inference_wrapper(image, prompt, mode):
    # Run core pipeline
    stylized_image, multiviews, camera_poses = run_promptfusion3d(
        prompt=prompt,
        input_image=image,
        mode=mode
    )

    # Export multiviews to disk for reconstruction (step done outside app for now)
    export_dir = "outputs/views"
    export_for_gaussian_splatting(export_dir, multiviews, camera_poses)

    # For now, assume your reconstruction creates outputs/scene.glb
    return stylized_image, multiviews[0], gr.HTML(
        "<iframe src='viewer.html' width='100%' height='512px' frameborder='0'></iframe>"
    )

with gr.Blocks() as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="e.g., a dog in lego style")
            mode = gr.Radio(choices=["Prompt Only", "Image + Prompt"], value="Prompt Only", label="Mode")
            image = gr.Image(type="pil", label="Upload Image (for 'Image + Prompt')")
            submit = gr.Button("Generate 3D Object")
        
        with gr.Column():
            stylized = gr.Image(label="Stylized Image (LPA)")
            view = gr.Image(label="One 3D View")
            viewer = gr.HTML()

    submit.click(fn=inference_wrapper, inputs=[image, prompt, mode], outputs=[stylized, view, viewer])

if __name__ == "__main__":
    demo.launch(share=True)
