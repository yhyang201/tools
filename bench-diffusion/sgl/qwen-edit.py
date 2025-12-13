import subprocess
import time
import requests
from io import BytesIO
import base64
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import wait_for_port, kill_process_tree
from openai import OpenAI

# Start the SGLang server
cmd = "sglang serve --model-path Qwen/Qwen-Image-Edit --port 30010"
process = subprocess.Popen(cmd, shell=True)

# Wait for the server to be ready
wait_for_port(host="127.0.0.1", port=30010)

# Initialize OpenAI client
client = OpenAI(
    api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1"
)

# Prepare image and prompt
image_url = "https://github.com/lm-sys/lm-sys.github.io/releases/download/test/TI2I_Qwen_Image_Edit_Input.jpg"
print(f"Downloading image from {image_url}...")
response = requests.get(image_url)
response.raise_for_status()
image_data = response.content
print("Image downloaded")

prompt = "Convert 2D style to 3D style."

times = []
for i in range(5):
    # Create a new BytesIO object for each request
    image_file = BytesIO(image_data)
    image_file.name = "input.jpg"  # Set filename for format detection

    start_time = time.time()
    response = client.images.edit(
        model="Qwen/Qwen-Image-Edit",
        image=image_file,
        prompt=prompt,
        n=1,
        size="1080x1920",
        response_format="b64_json",
    )
    end_time = time.time()
    
    elapsed = end_time - start_time
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.4f} seconds")

times = times[1:]
avg_time = sum(times) / len(times)
print(f"Average time: {avg_time:.4f} seconds")

# Save the result from the last run
if response.data:
    image_base64 = response.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    output_filename = "output_image_edit_sgl.png"
    with open(output_filename, "wb") as f:
        f.write(image_bytes)
    # print(f"Image saved to {os.path.abspath(output_filename)}")

kill_process_tree(process.pid)
