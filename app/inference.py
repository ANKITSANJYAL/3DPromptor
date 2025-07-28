import torch
import uuid
import tempfile
from PIL import Image
from app.prompt_parser import split_prompt_tokens
from app.models import load_models, generate_image_with_lpa
from app.zero123_wrapper import render_multiviews, generate_camera_poses
# from app.utils import save_video_from_frames

def run_promptfusion3d(prompt: str, input_image: Image.Image, mode: str) -> str:
    """
    Main orchestrator for PromptFusion3D.
    
    Args:
        prompt: Natural language prompt
        input_image: Optional input image (only used if mode is "Image + Prompt")
        mode: "Prompt Only" or "Image + Prompt"

    Returns:
        Path to turntable video (.mp4)
    """
    assert prompt, "Prompt must not be empty."

    # Load models (cached globally if needed)
    sdxl_pipe, zero123_pipe, clip_model = load_models()

    # Step 1: Prompt token separation
    object_tokens, style_tokens = split_prompt_tokens(prompt)

    # Step 2: Determine the base image to stylize
    if mode == "Prompt Only":
        base_image = generate_image_with_lpa(
            prompt=prompt,
            object_tokens=object_tokens,
            style_tokens=style_tokens,
            sdxl_pipeline=sdxl_pipe
        )
    elif mode == "Image + Prompt":
        if input_image is None:
            raise ValueError("Image must be provided for 'Image + Prompt' mode.")
        
        # Optional: add a captioning model to auto-describe the image
        base_image = generate_image_with_lpa(
            prompt=prompt,
            input_image=input_image,
            object_tokens=object_tokens,
            style_tokens=style_tokens,
            sdxl_pipeline=sdxl_pipe
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Step 3: Turntable view synthesis via Zero123
    # Step 3: Turntable view synthesis via Zero123
    turntable_frames = render_multiviews(
        input_image=base_image,
        model=zero123_pipe,
        num_views=24
    )

    # Step 4: Return everything to UI
    camera_poses = generate_camera_poses(num_views=24)
    return base_image, turntable_frames, camera_poses

