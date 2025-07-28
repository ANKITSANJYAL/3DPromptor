from diffusers import DiffusionPipeline
import torch
from PIL import Image

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-zero123", torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

image = Image.open("test_input.jpg").convert("RGB")
result = pipe(image=image, num_inference_steps=30, guidance_scale=4.0)
result.images[0].save("turntable_frame_1.png")
