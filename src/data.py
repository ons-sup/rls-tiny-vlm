"""
Dataset + collate function for ShapeScenes.

*** ADAPT THE LOADING SECTION BELOW ***
I don't have your exact generate_data.py output format in front of me.
Inspect what it actually writes to disk (a folder of images + a labels
file, or a single .pt/.npz tensor file) and adjust `load_split` below.
Everything after that point (Dataset, collate_fn) does not need to change.
"""

import torch
from torch.utils.data import Dataset
from .tokenizer import CharTokenizer


def load_split(data_dir: str, split: str):
    data = torch.load(f"{data_dir}/{split}.pt")
    return data["images"], data["words"]


def build_cache(data_dir: str, split: str, out_path: str):
    """
    Run this ONCE per split, then always load from out_path afterwards.
    This is the fix for 'only 2 CPU cores' in the brief: decode images once,
    not on every __getitem__ call.
    """
    images, words = load_split(data_dir, split)
    torch.save({"images": images, "words": words}, out_path)
    print(f"cached {split}: {images.shape[0]} examples -> {out_path}")


class ShapeScenesDataset(Dataset):
    def __init__(self, images: torch.Tensor, words: list[str], tokenizer: CharTokenizer):
        assert images.shape[0] == len(words)
        self.images = images          # (N, 3, 64, 64) uint8
        self.words = words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0          # -> [0, 1] float
        ids = torch.tensor(self.tokenizer.encode(self.words[idx]), dtype=torch.long)
        return img, ids


def collate_fn(batch, eos_id: int):
    """
    Batches variable-length words.

    Teacher forcing setup:
      full sequence per example = [c0, c1, ..., c_{L-1}, eos]   (length L+1)
      input_letters  = full[:-1]   (length L)   fed into the decoder
      targets        = full[1:]    (length L)   what each position should predict

    Padding: pad input_letters with eos_id (never used meaningfully - the
    corresponding target position is masked out of the loss anyway).
    Targets are padded with -100, which nn.CrossEntropyLoss ignores by default.
    """
    imgs, id_seqs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)                       # (B, 3, 64, 64)

    inputs = [seq[:-1] for seq in id_seqs]                # word letters only
    targets = [seq[1:] for seq in id_seqs]                # shifted by one
    lengths = [len(s) for s in inputs]
    max_len = max(lengths)
    B = len(inputs)

    input_pad = torch.full((B, max_len), eos_id, dtype=torch.long)
    target_pad = torch.full((B, max_len), -100, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        L = len(inp)
        input_pad[i, :L] = inp
        target_pad[i, :L] = tgt

    return imgs, input_pad, target_pad, torch.tensor(lengths)
