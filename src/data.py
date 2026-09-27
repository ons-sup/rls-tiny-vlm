"""
PyTorch Dataset and collate function for ShapeScenes.

generate_data.py sauvegarde chaque split comme un fichier .pt contenant
un dict avec deux clés : "images" (tensor uint8) et "words" (liste de str).
"""

import torch
from torch.utils.data import Dataset
from .tokenizer import CharTokenizer


def load_split(data_dir: str, split: str):
    data = torch.load(f"{data_dir}/{split}.pt")
    return data["images"], data["words"]


class ShapeScenesDataset(Dataset):
    def __init__(self, images: torch.Tensor, words: list, tokenizer: CharTokenizer):
        assert images.shape[0] == len(words)
        self.images = images          # (N, 3, 64, 64) uint8
        self.words = words
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        ids = torch.tensor(self.tokenizer.encode(self.words[idx]), dtype=torch.long)
        return img, ids


def collate_fn(batch, eos_id: int):
    imgs, id_seqs = zip(*batch)
    imgs = torch.stack(imgs, dim=0)

    inputs = [seq[:-1] for seq in id_seqs]
    targets = [seq[1:] for seq in id_seqs]
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