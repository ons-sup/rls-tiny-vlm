"""
S1 - required benchmark: training throughput (images/sec) at several batch sizes.
Handles the three traps from the brief: warm-up, async GPU execution, data loading.

Run: python -m benchmarks.throughput
"""

import time
import torch
import torch.nn as nn

from src.model.vlm import TinyVLM


def benchmark(model, device, batch_sizes=(1, 16, 64, 256), warmup_steps=5, timed_steps=20):
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    results = []

    for bs in batch_sizes:
        images = torch.randn(bs, 3, 64, 64, device=device)
        letters = torch.randint(0, 26, (bs, 20), device=device)   # fixed length for a clean timing
        targets = torch.randint(0, 27, (bs, 20), device=device)

        # warm-up: first iterations pay for kernel compilation / cuDNN
        # autotuning / memory allocator warm-up - don't include them in the timing
        for _ in range(warmup_steps):
            logits = model(images, letters)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 27), targets.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()

        if device.type == "cuda":
            torch.cuda.synchronize()   # GPU calls return before work finishes - must sync before timing
        start = time.perf_counter()
        for _ in range(timed_steps):
            logits = model(images, letters)
            loss = nn.functional.cross_entropy(logits.reshape(-1, 27), targets.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        images_per_sec = (bs * timed_steps) / elapsed
        results.append((bs, images_per_sec))
        print(f"batch_size={bs:4d}  {images_per_sec:8.1f} images/sec")

    return results


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    model = TinyVLM(vocab_size=27, d_model=128, num_heads=4, num_layers=2, num_visual_tokens=64)
    benchmark(model, device)
    # NOTE: this times the model + optimizer step only (no real DataLoader),
    # so it isolates compute from data loading. State that choice explicitly
    # in your report, and repeat this a few times to report mean and spread.
