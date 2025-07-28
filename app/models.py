import torch
from typing import List, Optional
from PIL import Image
from app.utils import inject_lpa_attention
from diffusers import StableDiffusionXLPipeline
from app.zero123_wrapper import load_zero123_model, render_multiviews


# Global cache
_cached = {"sdxl": None, "zero123": None}

def load_models():
    """
    Loads SDXL and Zero123 pipelines
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # SDXL
    if _cached["sdxl"] is None:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)
        _cached["sdxl"] = pipe

    # Zero123
    if _cached["zero123"] is None:
        model = load_zero123_model(device)
        _cached["zero123"] = model

    return _cached["sdxl"], _cached["zero123"], None  # No CLIP needed


def generate_image_with_lpa(
    prompt: str,
    object_tokens: List[str],
    style_tokens: List[str],
    sdxl_pipeline,
    input_image: Optional[Image.Image] = None,
) -> Image.Image:
    """
    Generate stylized image using LPA + SDXL
    """
    object_prompt = " ".join(object_tokens)
    style_prompt = " ".join(style_tokens)
    final_prompt = f"{object_prompt} in {style_prompt}" if style_prompt else object_prompt

    inject_lpa_attention(sdxl_pipeline, object_tokens, style_tokens)

    result = sdxl_pipeline(
        prompt=final_prompt,
        num_inference_steps=50,
        guidance_scale=7.5,
    )
    return result.images[0]


def synthesize_3d_views(image: Image.Image, zero123_model, num_views: int = 24) -> List[Image.Image]:
    """
    Generate novel views using Zero123-XL
    """
    return render_multiviews(image, zero123_model, num_views=num_views)
