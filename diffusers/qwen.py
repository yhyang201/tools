from diffusers import DiffusionPipeline
import torch
import time


model_name = "Qwen/Qwen-Image"

# Load the pipeline
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16
    device = "cuda"
else:
    torch_dtype = torch.float32
    device = "cpu"

pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch_dtype)
pipe = pipe.to(device)

# Generate image
prompt = 'A curious raccoon'

negative_prompt = " " # using an empty string if you do not have specific concept to remove


times = []
image = None
for i in range(5):
    start_time = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=720,
        height=720,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device=device).manual_seed(42)
    ).images[0]
    end_time = time.time()
    elapsed = end_time - start_time
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.4f} seconds")

times = times[1:]
avg_time = sum(times) / len(times)
print(f"Average time: {avg_time:.4f} seconds")

image.save("example.png")