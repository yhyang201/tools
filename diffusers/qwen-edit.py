import os
from PIL import Image
import torch
import requests
from io import BytesIO

from diffusers import QwenImageEditPipeline

pipeline = QwenImageEditPipeline.from_pretrained("Qwen/Qwen-Image-Edit")
print("pipeline loaded")
pipeline.to(torch.bfloat16)
pipeline.to("cuda")
pipeline.set_progress_bar_config(disable=None)
image_url = "https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg"

# Download image
print(f"Downloading image from {image_url}...")
response = requests.get(image_url)
response.raise_for_status()
image = Image.open(BytesIO(response.content)).convert("RGB")
print("Image downloaded and opened")
prompt = "Change the rabbit's color to purple, with a flash light background."
inputs = {
    "image": image,
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 50,
}

with torch.inference_mode():
    import time

    times = []
    output = None
    for i in range(5):
        start_time = time.time()
        output = pipeline(**inputs)
        end_time = time.time()
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"Run {i+1}: {elapsed:.4f} seconds")
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.4f} seconds")
    output_image = output.images[0]
    output_image.save("output_image_edit.png")
    print("image saved at", os.path.abspath("output_image_edit.png"))