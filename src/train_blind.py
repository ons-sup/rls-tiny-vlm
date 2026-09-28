"""
E1 - Blind baseline: train the SAME model architecture, but with images
replaced by zeros. If the model still does well, it means it was mostly
guessing from letter statistics, not really looking at the image.
Compare this script's final accuracy to your normal run_eval.py result.

Usage:
  python -m src.train_blind --epochs 10
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .tokenizer import CharTokenizer
from .data import ShapeScenesDataset, collate_fn, load_split
from .model.vlm import TinyVLM


class BlindDataset(ShapeScenesDataset):
    """Same as ShapeScenesDataset, but always returns a zeroed-out image."""
    def __getitem__(self, idx):
        img, ids = super().__getitem__(idx)
        return torch.zeros_like(img), ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    model = TinyVLM(vocab_size=tokenizer.vocab_size).to(device)

    train_images, train_words = load_split(args.data_dir, "train")
    val_images, val_words = load_split(args.data_dir, "val")

    train_ds = BlindDataset(train_images, train_words, tokenizer)
    val_ds = BlindDataset(val_images, val_words, tokenizer)
    collate = lambda b: collate_fn(b, eos_id=tokenizer.eos_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for images, inputs, targets, _ in train_loader:
            images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)
            logits = model(images, inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batches += 1

        model.eval()
        v_loss, v_batches = 0.0, 0
        with torch.no_grad():
            for images, inputs, targets, _ in val_loader:
                images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)
                logits = model(images, inputs)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
                )
                v_loss += loss.item(); v_batches += 1

        print(f"epoch {epoch:3d}  train {epoch_loss/n_batches:.4f}  val {v_loss/v_batches:.4f}")
        torch.save(model.state_dict(), f"checkpoint_blind_epoch{epoch}.pt")