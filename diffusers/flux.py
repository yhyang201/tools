import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("cuda")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A curious raccoon"
import time

times = []
image = None
for i in range(5):
    start_time = time.time()
    image = pipe(
        prompt,
        height=720,
        width=720,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    end_time = time.time()
    elapsed = end_time - start_time
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.4f} seconds")

times = times[1:]
avg_time = sum(times) / len(times)
print(f"Average time: {avg_time:.4f} seconds")
image.save("flux-dev.png")