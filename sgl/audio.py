import time
from openai import OpenAI

client = OpenAI(api_key="sk-123456", base_url="http://localhost:30000/v1")
audio_file_path = "https://raw.githubusercontent.com/yhyang201/tools/main/test_files/audio_file.mp3"

start_time = time.perf_counter()

response = client.chat.completions.create(
    model="default",
    messages=[{
        "role": "user",
        "content": [
            {"type": "audio_url", "audio_url": {"url": audio_file_path}},
            {"type": "text", "text": "Please transcribe this audio into text"},
        ],
    }],
    temperature=0,
    max_tokens=1024,
    stream=True, 
)

first_token_time = 0
token_count = 0
content = ""

for chunk in response:
    delta = chunk.choices[0].delta.content
    if delta:
        if first_token_time == 0:
            first_token_time = time.perf_counter()
        token_count += 1
        content += delta

end_time = time.perf_counter()

print(content)
print("-" * 20)

if first_token_time > 0:
    ttft = (first_token_time - start_time) * 1000
    decode_time = end_time - first_token_time
    tpot = (decode_time / (token_count - 1)) * 1000 if token_count > 1 else 0
    
    print(f"TTFT: {ttft:.2f} ms")
    print(f"TPOT: {tpot:.2f} ms")
else:
    print("No tokens generated.")
