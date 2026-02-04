import time
import argparse
from tokenizers import Tokenizer
import os

def profile_tokenizer(tokenizer_path, text_path):
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Reading text from: {text_path}")
    with open(text_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Profiling {len(lines)} lines")

    # End-to-end Single (sequential)
    start_time = time.perf_counter()
    for line in lines:
        tokenizer.encode(line)
    end_time = time.perf_counter()
    duration_single = end_time - start_time
    avg_single = (duration_single / len(lines)) * 1e9

    # End-to-end Batch (parallel)
    start_time = time.perf_counter()
    tokenizer.encode_batch(lines)
    end_time = time.perf_counter()
    duration_batch = end_time - start_time
    avg_batch = (duration_batch / len(lines)) * 1e9

    print("Python Profiling Results (Average per line)")
    print(f"Single encode: {avg_single:>10.2f} ns")
    print(f"Batch encode:  {avg_batch:>10.2f} ns")
    print(f"Speedup ratio: {duration_single / duration_batch:>10.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    profile_tokenizer(args.tokenizer, args.text)
