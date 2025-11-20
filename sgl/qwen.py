import subprocess
import time
from ..utils import wait_for_port
from openai import OpenAI

cmd = "sglang serve --model-path Qwen/Qwen-Image --port 30010"

process = subprocess.Popen(cmd, shell=True)

wait_for_port(host="127.0.0.1", port=30010)

client = OpenAI(
        api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1"
    )

times = []
for i in range(5):
    start_time = time.time()
    content = client.images.generate(
        prompt="A curious raccoon",
        n=1,
        size="720x720"
    )
    end_time = time.time()
    elapsed = end_time - start_time
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.4f} seconds")

avg_time = sum(times) / len(times)
print(f"Average time: {avg_time:.4f} seconds")