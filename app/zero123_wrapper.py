import os
import sys
import torch
import yaml
import numpy as np
from PIL import Image
from typing import List
from torchvision import transforms

# Set up path to zero123 repo and import config loader
ZERO123_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "zero123", "zero123"))
sys.path.append(ZERO123_ROOT)

from ldm.util import instantiate_from_config

def load_zero123_model(device="cuda"):
    """
    Loads the Zero123-XL model using config and checkpoint.
    """
    config_path = "/Users/ankitsanjyal/Desktop/Projects/3DPromptor/zero123/zero123/configs/sd-objaverse-finetune-c_concat-256.yaml"
    ckpt_path = "/Users/ankitsanjyal/Desktop/Projects/3DPromptor/zero123/zero123/checkpoints/zero123-xl.ckpt"


    print(f"[📦] Loading Zero123 from: {ckpt_path}")

    # Load model config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = instantiate_from_config(config["model"])
    model.to(device)

    # Load checkpoint weights
    import torch.serialization
    from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

    with torch.serialization.safe_globals([ModelCheckpoint]):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["state_dict"]


    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model


def generate_camera_poses(num_views: int = 24):
    """
    Returns evenly spaced 360-degree azimuth poses with fixed elevation.
    """
    azimuths = np.linspace(0, 360, num_views, endpoint=False)
    elevations = np.zeros_like(azimuths)  # frontal view
    return [{"theta": np.deg2rad(az), "phi": np.deg2rad(el), "radius": 1.0}
            for az, el in zip(azimuths, elevations)]


def render_multiviews(input_image: Image.Image, model, num_views: int = 24) -> List[Image.Image]:
    """
    Renders novel views from a stylized image using Zero123-XL.
    """
    device = next(model.parameters()).device

    # Resize and convert image
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = preprocess(input_image).unsqueeze(0).to(device)

    # Get conditioning from input image
    with torch.no_grad():
        conditioning = model.get_learned_conditioning(image_tensor)

    # Create views
    camera_poses = generate_camera_poses(num_views)
    views = []

    for pose in camera_poses:
        with torch.no_grad():
            latent = model.sample(c=conditioning, camera=pose)
            decoded = model.decode_first_stage(latent)
            image_out = transforms.ToPILImage()(decoded.squeeze(0).cpu().clamp(0, 1))
            views.append(image_out)

    return views
