"""
Training loop + E0 (overfit one batch - the bug check, not a result).

Usage:
  python -m src.train --mode e0        # sanity check: can the pipeline learn at all?
  python -m src.train --mode full      # real training run
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .tokenizer import CharTokenizer
from .data import ShapeScenesDataset, collate_fn, load_split
from .model.vlm import TinyVLM


def get_device(flag: str = "auto") -> torch.device:
    if flag == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(flag)


def make_loader(images, words, tokenizer, batch_size, shuffle):
    ds = ShapeScenesDataset(images, words, tokenizer)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, eos_id=tokenizer.eos_id),
    )


def run_e0(model, loader, device, steps=300, lr=3e-4):
    """
    Overfit exactly one batch to near-zero loss.
    If this doesn't work, nothing else in the pipeline is worth debugging yet -
    fix this first.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    images, inputs, targets, _ = next(iter(loader))
    images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)

    losses = []
    for step in range(steps):
        logits = model(images, inputs)                       # (B, L, vocab)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"E0 step {step:4d}  loss {loss.item():.4f}")

    print(f"E0 final loss: {losses[-1]:.4f} (should be close to 0)")
    return losses   # plot this list - required deliverable for Level 1


def run_full_training(model, train_loader, val_loader, device, epochs=10, lr=3e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        for images, inputs, targets, _ in train_loader:
            images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)
            logits = model(images, inputs)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / n_batches)

        model.eval()
        v_loss, v_batches = 0.0, 0
        with torch.no_grad():
            for images, inputs, targets, _ in val_loader:
                images, inputs, targets = images.to(device), inputs.to(device), targets.to(device)
                logits = model(images, inputs)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
                )
                v_loss += loss.item()
                v_batches += 1
        val_losses.append(v_loss / v_batches)

        print(f"epoch {epoch:3d}  train {train_losses[-1]:.4f}  val {val_losses[-1]:.4f}")
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pt")   # push to Drive/GitHub

    return train_losses, val_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["e0", "full"], default="e0")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    tokenizer = CharTokenizer()
    model = TinyVLM(vocab_size=tokenizer.vocab_size).to(device)

    train_images, train_words = load_split(args.data_dir, "train")
    train_loader = make_loader(train_images, train_words, tokenizer, args.batch_size, shuffle=True)

    if args.mode == "e0":
        run_e0(model, train_loader, device)
    else:
        val_images, val_words = load_split(args.data_dir, "val")
        val_loader = make_loader(val_images, val_words, tokenizer, args.batch_size, shuffle=False)
        run_full_training(model, train_loader, val_loader, device, epochs=args.epochs)
