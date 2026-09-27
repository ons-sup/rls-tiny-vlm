"""
Character-level tokenizer: 26 letters (a-z) + <eos>.
No space token, no padding token in the vocabulary itself -
padding is handled at the batching level (see data.py).
"""

LETTERS = list("abcdefghijklmnopqrstuvwxyz")
EOS = "<eos>"


class CharTokenizer:
    def __init__(self):
        self.stoi = {c: i for i, c in enumerate(LETTERS)}   # a->0 ... z->25
        self.stoi[EOS] = len(LETTERS)                        # <eos> -> 26
        self.itos = {i: c for c, i in self.stoi.items()}
        self.eos_id = self.stoi[EOS]
        self.vocab_size = len(self.stoi)                     # 27

    def encode(self, word: str) -> list[int]:
        """'cat' -> [c_id, a_id, t_id, eos_id]"""
        ids = [self.stoi[c] for c in word.lower()]
        ids.append(self.eos_id)
        return ids

    def decode(self, ids) -> str:
        """Stops at the first <eos>. Ignores anything after it."""
        chars = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            chars.append(self.itos.get(i, "?"))
        return "".join(chars)


if __name__ == "__main__":
    # quick sanity check - run: python -m src.tokenizer
    tok = CharTokenizer()
    ids = tok.encode("cat")
    print("encoded:", ids)
    print("decoded:", tok.decode(ids))
    assert tok.decode(ids) == "cat"
    print("vocab_size:", tok.vocab_size)
