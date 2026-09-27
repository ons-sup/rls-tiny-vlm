"""
Sanity check on tensor shapes end to end (image -> logits) and on
generation output. Run: python -m tests.test_shapes
"""

import torch
from src.model.vlm import TinyVLM


def test_forward_shapes():
    model = TinyVLM(vocab_size=27, d_model=128, num_heads=4, num_layers=2,
                     num_visual_tokens=64, max_letters=45)
    images = torch.randn(3, 3, 64, 64)
    letter_inputs = torch.randint(0, 26, (3, 10))   # pretend 10-letter prefix
    logits = model(images, letter_inputs)
    assert logits.shape == (3, 10, 27), logits.shape
    print("forward pass shape OK:", logits.shape)


def test_generate_stops():
    model = TinyVLM(vocab_size=27, d_model=128, num_heads=4, num_layers=2,
                     num_visual_tokens=64, max_letters=45)
    images = torch.randn(2, 3, 64, 64)
    out = model.generate(images)
    assert len(out) == 2
    for seq in out:
        assert len(seq) <= 45
    print("generate() OK, lengths:", [len(s) for s in out])


if __name__ == "__main__":
    test_forward_shapes()
    test_generate_stops()
