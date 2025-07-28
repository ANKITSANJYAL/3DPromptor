import os
from PIL import Image
from typing import List
import numpy as np
import torch


def export_for_gaussian_splatting(output_dir: str, images: List[Image.Image], poses: List[tuple]):
    """
    Saves images and camera poses in a format compatible with Gaussian Splatting repo.
    Assumes:
        - images: list of PIL.Image of novel views
        - poses: list of (azimuth, elevation) in degrees
    """
    os.makedirs(output_dir, exist_ok=True)
    intrinsics_path = os.path.join(output_dir, "intrinsics.txt")
    poses_path = os.path.join(output_dir, "poses.txt")

    # Simple intrinsics (assumes 256x256 images)
    with open(intrinsics_path, "w") as f:
        f.write("256 256\n")           # width height
        f.write("128.0 128.0 128.0\n") # fx fy cx (assuming centered)
        f.write("0.0\n")               # radial distortion

    with open(poses_path, "w") as f:
        for idx, (img, (az, el)) in enumerate(zip(images, poses)):
            fname = f"view_{idx:03}.png"
            img.save(os.path.join(output_dir, fname))
            # Write dummy poses (real camera matrices come from COLMAP if needed)
            f.write(f"{fname} {az} {el} 1.0\n")  # placeholder: az, el, scale

    print(f"[✔] Exported {len(images)} images to {output_dir} for 3D reconstruction.")


def inject_lpa_attention(pipe, object_tokens: List[str], style_tokens: List[str], boundary_timestep: int = 35):
    """
    Modifies SDXL pipeline cross-attention to apply object tokens early, style tokens late.
    Works by monkey-patching attention layer forward passes.
    """
    tokenizer = pipe.tokenizer_2 if hasattr(pipe, 'tokenizer_2') else pipe.tokenizer
    obj_ids = tokenizer(object_tokens, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()
    style_ids = tokenizer(style_tokens, add_special_tokens=False, return_tensors="pt").input_ids[0].tolist()

    def patch_attention(attn_module, block_type, timestep):

        def modified_forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            # Determine which tokens to allow
            if encoder_hidden_states is not None:
                input_ids = encoder_hidden_states.argmax(dim=-1).tolist() if encoder_hidden_states.dim() == 3 else []
                if block_type == 'down' and timestep < boundary_timestep:
                    # Keep object token attention only
                    mask = [1 if idx in obj_ids else 0 for idx in input_ids]
                elif block_type in {'mid', 'up'} and timestep >= boundary_timestep:
                    # Keep style token attention only
                    mask = [1 if idx in style_ids else 0 for idx in input_ids]
                else:
                    mask = [1] * len(input_ids)

                attention_mask = torch.tensor(mask, dtype=torch.bool, device=encoder_hidden_states.device).unsqueeze(0).unsqueeze(1)

            return attn_module._original_forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )

        return modified_forward

    # Hook into U-Net blocks
    unet = pipe.unet
    timestep = 0  # default for now

    for i, block in enumerate(unet.down_blocks):
        for attn in getattr(block, "attentions", []):
            if not hasattr(attn, "_original_forward"):
                attn._original_forward = attn.forward
                attn.forward = patch_attention(attn, "down", timestep)

    for i, attn in enumerate(getattr(unet, "mid_block").attentions):
        if not hasattr(attn, "_original_forward"):
            attn._original_forward = attn.forward
            attn.forward = patch_attention(attn, "mid", timestep)

    for i, block in enumerate(unet.up_blocks):
        for attn in getattr(block, "attentions", []):
            if not hasattr(attn, "_original_forward"):
                attn._original_forward = attn.forward
                attn.forward = patch_attention(attn, "up", timestep)
