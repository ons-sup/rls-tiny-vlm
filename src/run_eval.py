"""
Load a trained checkpoint and run evaluation on a data split.

Usage:
  python -m src.run_eval --checkpoint checkpoint_epoch9.pt --split test
"""

import argparse
import torch
from torch.utils.data import DataLoader

from .tokenizer import CharTokenizer
from .data import ShapeScenesDataset, collate_fn, load_split
from .model.vlm import TinyVLM
from .evaluate import run_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()

    model = TinyVLM(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    images, words = load_split(args.data_dir, args.split)
    ds = ShapeScenesDataset(images, words, tokenizer)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, eos_id=tokenizer.eos_id),
    )

    results = run_evaluation(model, loader, tokenizer, device)
    print("Exact match:", results["exact_match"])
    print("Attribute accuracy:")
    for k, v in results["attribute_accuracy"].items():
        if v is not None:
            print(f"  {k}: {v:.4f}")