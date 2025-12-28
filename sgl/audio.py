import time
import asyncio
import statistics
import math
from openai import AsyncOpenAI

API_KEY = "sk-123456"
BASE_URL = "http://localhost:30000/v1"
AUDIO_FILE_PATH = "https://raw.githubusercontent.com/yhyang201/tools/main/test_files/audio_file.mp3"
CONCURRENT_REQUESTS = 128  
HISTOGRAM_BINS = 10

async def send_request(client, request_id):
    start_time = time.perf_counter()
    first_token_time = 0
    token_count = 0
    
    try:
        response = await client.chat.completions.create(
            model="default",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": AUDIO_FILE_PATH}},
                    {"type": "text", "text": "Please transcribe this audio into text"},
                ],
            }],
            temperature=0,
            max_tokens=16,
            stream=True, 
        )

        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token_time == 0:
                    first_token_time = time.perf_counter()
                token_count += 1
        
        end_time = time.perf_counter()

        if first_token_time > 0:
            ttft = (first_token_time - start_time) * 1000
            decode_time = end_time - first_token_time
            if token_count > 1:
                tpot = (decode_time / (token_count - 1)) * 1000
            else:
                tpot = 0.0
            return True, ttft, tpot, token_count, None
        else:
            return False, 0, 0, 0, "No tokens received"

    except Exception as e:
        return False, 0, 0, 0, str(e)

def print_histogram(data, label, unit="ms", bins=10):
    if not data:
        print(f"\nNo data for {label}.")
        return

    min_val = min(data)
    max_val = max(data)
    total = len(data)
    
    if max_val == min_val:
        print(f"\n{label} Distribution (Total: {total})")
        print(f"{min_val:.2f}{unit} : {'▇' * 50} ({total})")
        return

    step = (max_val - min_val) / bins
    
    bucket_counts = [0] * bins
    
    for x in data:
        idx = int((x - min_val) / step)
        if idx >= bins:
            idx = bins - 1
        bucket_counts[idx] += 1

    print(f"\n{label} Distribution (N={total}, Min={min_val:.2f}{unit}, Max={max_val:.2f}{unit})")
    print("-" * 60)

    max_count = max(bucket_counts)
    scale_factor = 50 / max_count if max_count > 0 else 1

    for i in range(bins):
        range_start = min_val + (i * step)
        range_end = min_val + ((i + 1) * step)
        count = bucket_counts[i]
        bar_len = int(count * scale_factor)
        bar = '▇' * bar_len
        
        range_str = f"{range_start:7.1f}-{range_end:7.1f}"
        print(f"{range_str} : {bar:<50} ({count})")
    print("-" * 60)

def calculate_stats(data):
    if not data: return 0, 0, 0, 0
    avg_val = statistics.mean(data)
    min_val = min(data)
    max_val = max(data)
    sorted_data = sorted(data)
    p99_val = sorted_data[min(int(len(sorted_data) * 0.99), len(sorted_data) - 1)]
    return avg_val, min_val, max_val, p99_val

async def main():
    print(f"Starting {CONCURRENT_REQUESTS} concurrent requests...")
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    tasks = [send_request(client, i) for i in range(CONCURRENT_REQUESTS)]
    
    results = await asyncio.gather(*tasks)

    ttft_list = []
    tpot_list = []
    success_count = 0
    fail_count = 0
    errors = []

    for success, ttft, tpot, count, err in results:
        if success:
            success_count += 1
            ttft_list.append(ttft)
            if count > 1:
                tpot_list.append(tpot)
        else:
            fail_count += 1
            if err: errors.append(err)

    ttft_stats = calculate_stats(ttft_list)
    tpot_stats = calculate_stats(tpot_list)

    print("\n" + "=" * 65)
    print(f"{'Metric':<10} | {'Avg':<10} | {'Best':<10} | {'Worst':<10} | {'P99':<10}")
    print("-" * 65)
    print(f"{'TTFT (ms)':<10} | {ttft_stats[0]:<10.2f} | {ttft_stats[1]:<10.2f} | {ttft_stats[2]:<10.2f} | {ttft_stats[3]:<10.2f}")
    print(f"{'TPOT (ms)':<10} | {tpot_stats[0]:<10.2f} | {tpot_stats[1]:<10.2f} | {tpot_stats[2]:<10.2f} | {tpot_stats[3]:<10.2f}")
    print("=" * 65)

    print_histogram(ttft_list, "TTFT", bins=10)
    print_histogram(tpot_list, "TPOT", bins=10)
    
    print(f"\nTotal: {CONCURRENT_REQUESTS}, Success: {success_count}, Failed: {fail_count}")

if __name__ == "__main__":
    asyncio.run(main())
