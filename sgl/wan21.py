import subprocess
import time
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import wait_for_port, kill_process_tree
from openai import OpenAI

cmd = "sglang serve --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers --port 30010"

process = subprocess.Popen(cmd, shell=True)

wait_for_port(host="127.0.0.1", port=30010)

client = OpenAI(
    api_key="sk-proj-1234567890", base_url="http://localhost:30010/v1"
)

def wait_for_video_completion(client, video_id, timeout=300, check_interval=3):
    start = time.time()
    video = client.videos.retrieve(video_id)

    while video.status not in ("completed", "failed"):
        time.sleep(check_interval)
        video = client.videos.retrieve(video_id)
        assert time.time() - start < timeout, "video generate timeout"

    return video

def _create_wait_and_download(
    client: OpenAI, prompt: str, size: str
) -> bytes:
    video = client.videos.create(prompt=prompt, size=size)
    video_id = video.id

    video = wait_for_video_completion(client, video_id, timeout=300)
    assert video.status == "completed", "video generate failed"

    response = client.videos.download_content(
        video_id=video_id,
    )
    content = response.read()
    return content

times = []
for i in range(5):
    start_time = time.time()
    content = _create_wait_and_download(
        client, "A cat walks on the grass, realistic", "832x480"
    )
    end_time = time.time()
    elapsed = end_time - start_time
    times.append(elapsed)
    print(f"Run {i+1}: {elapsed:.4f} seconds")

avg_time = sum(times) / len(times)
print(f"Average time: {avg_time:.4f} seconds")
kill_process_tree(process.pid)
