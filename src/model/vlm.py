"""
Full model: image -> visual tokens -> [visual tokens + letter tokens] ->
decoder -> next-letter logits -> greedy decoding.
"""

import torch
import torch.nn as nn

from .encoder import CNNEncoder
from .decoder import Decoder
from .attention import build_prefix_mask


class TinyVLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 27,          # 26 letters + <eos>
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        num_visual_tokens: int = 64,   # 8*8 from the CNN, must match encoder output
        max_letters: int = 45,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.max_letters = max_letters
        self.eos_id = vocab_size - 1

        self.vision_encoder = CNNEncoder(d_model=d_model)
        self.letter_embed = nn.Embedding(vocab_size, d_model)
        self.decoder = Decoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            max_seq_len=num_visual_tokens + max_letters,
            dropout=dropout,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, images: torch.Tensor, letter_inputs: torch.Tensor) -> torch.Tensor:
        """
        images:        (B, 3, 64, 64)
        letter_inputs: (B, L) token ids fed with teacher forcing
        returns:       (B, L, vocab_size) logits - ONLY for the letter positions
        """
        visual_tokens = self.vision_encoder(images)          # (B, V, d_model)
        letter_tokens = self.letter_embed(letter_inputs)      # (B, L, d_model)

        x = torch.cat([visual_tokens, letter_tokens], dim=1)  # (B, V+L, d_model)
        V, L = visual_tokens.shape[1], letter_tokens.shape[1]
        mask = build_prefix_mask(V, L).to(images.device)

        out = self.decoder(x, mask)                           # (B, V+L, d_model)
        letter_out = out[:, V - 1:, :]                          # last visual token jusqu'à la fin : L+1 positions
        logits = self.head(letter_out)                          # (B, L+1, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, images: torch.Tensor) -> list[list[int]]:
        """
        Greedy decoding, one letter per step, until <eos> or max_letters.
        images: (B, 3, 64, 64)
        returns: list of token-id lists (one per image, includes trailing <eos> if reached)
        """
        self.eval()
        B = images.shape[0]
        device = images.device
        visual_tokens = self.vision_encoder(images)   # (B, V, d_model)
        V = visual_tokens.shape[1]

        generated = [[] for _ in range(B)]
        done = [False] * B
        letter_ids = torch.zeros((B, 0), dtype=torch.long, device=device)

        for _ in range(self.max_letters):
            # Re-running the decoder on the whole sequence at each step is
            # simple and correct but wastes compute (no KV cache). Fine for a
            # model this size; a KV cache would be a good Level 3 extension.
            L = letter_ids.shape[1]
            letter_tokens = self.letter_embed(letter_ids) if L > 0 else \
                torch.zeros((B, 0, visual_tokens.shape[-1]), device=device)
            x = torch.cat([visual_tokens, letter_tokens], dim=1)   # (B, V+L, d_model)
            mask = build_prefix_mask(V, L).to(device)              # (V+L, V+L)

            out = self.decoder(x, mask)
            next_logits = self.head(out[:, -1, :])       # (B, vocab_size) - predicts next letter
            next_id = next_logits.argmax(dim=-1)          # (B,)

            letter_ids = torch.cat([letter_ids, next_id.unsqueeze(1)], dim=1)
            for b in range(B):
                if not done[b]:
                    generated[b].append(next_id[b].item())
                    if next_id[b].item() == self.eos_id:
                        done[b] = True
            if all(done):
                break

        return generated
