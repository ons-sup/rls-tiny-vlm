"""
Evaluation: exact-match accuracy + per-attribute accuracy.

The parser must survive broken/misspelled outputs like "largeredcirle"
(missing a letter). Strategy: for each expected slot (size, color, shape,
[relation, size, color, shape]), pick the candidate from the known vocabulary
with the smallest edit distance to the remaining prefix of the string,
then consume that many characters and move on. This is forgiving of small
spelling mistakes without needing a hand-built grammar.
"""

import torch

SIZES = ["small", "large"]
COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["circle", "square", "triangle", "cross"]
RELATIONS = ["leftof", "rightof", "above", "below"]


def edit_distance(a: str, b: str) -> int:
    # classic O(len(a)*len(b)) DP - fine at this scale (words <= 45 chars)
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def best_match(remaining: str, candidates: list[str]) -> str:
    """Pick the candidate whose length-matched prefix is closest to `remaining`."""
    scored = []
    for c in candidates:
        prefix = remaining[: len(c)]
        scored.append((edit_distance(prefix, c), c))
    scored.sort(key=lambda t: t[0])
    return scored[0][1]


def parse_word(word: str) -> dict:
    """
    Returns a dict with keys: size, color, shape, and (if a second object is
    present) relation, size2, color2, shape2. Missing/unparseable slots are None.
    """
    result = {"size": None, "color": None, "shape": None,
              "relation": None, "size2": None, "color2": None, "shape2": None}
    s = word
    try:
        size = best_match(s, SIZES); s = s[len(size):]
        result["size"] = size
        color = best_match(s, COLORS); s = s[len(color):]
        result["color"] = color
        shape = best_match(s, SHAPES); s = s[len(shape):]
        result["shape"] = shape
        if len(s) > 0:   # a second object follows
            relation = best_match(s, RELATIONS); s = s[len(relation):]
            result["relation"] = relation
            size2 = best_match(s, SIZES); s = s[len(size2):]
            result["size2"] = size2
            color2 = best_match(s, COLORS); s = s[len(color2):]
            result["color2"] = color2
            shape2 = best_match(s, SHAPES); s = s[len(shape2):]
            result["shape2"] = shape2
    except Exception:
        pass   # leave remaining slots as None on total failure
    return result


def evaluate(predictions: list[str], references: list[str]) -> dict:
    """
    predictions/references: decoded word strings, same length, same order.
    Returns exact-match accuracy and per-attribute accuracy.
    """
    assert len(predictions) == len(references)
    n = len(predictions)
    exact_correct = 0
    attr_correct = {k: 0 for k in ["size", "color", "shape", "relation", "size2", "color2", "shape2"]}
    attr_total = {k: 0 for k in attr_correct}

    for pred, ref in zip(predictions, references):
        if pred == ref:
            exact_correct += 1
        p_parsed = parse_word(pred)
        r_parsed = parse_word(ref)
        for key in attr_correct:
            if r_parsed[key] is not None:   # only count attributes that truly exist in the reference
                attr_total[key] += 1
                if p_parsed[key] == r_parsed[key]:
                    attr_correct[key] += 1

    attr_acc = {k: (attr_correct[k] / attr_total[k] if attr_total[k] > 0 else None)
                for k in attr_correct}
    return {
        "exact_match": exact_correct / n,
        "attribute_accuracy": attr_acc,
        "n_examples": n,
    }


@torch.no_grad()
def run_evaluation(model, loader, tokenizer, device):
    model.eval()
    all_preds, all_refs = [], []
    for images, _, targets, lengths in loader:
        images = images.to(device)
        generated_ids = model.generate(images)
        for ids, tgt_len in zip(generated_ids, lengths):
            all_preds.append(tokenizer.decode(ids))
        # rebuild references from targets (targets = shifted letters + eos, ignore -100 pads)
        for row in targets:
            valid = row[row != -100].tolist()
            all_refs.append(tokenizer.decode(valid))
    return evaluate(all_preds, all_refs)


if __name__ == "__main__":
    # quick sanity check of the parser alone, no model needed
    examples = ["largeredcircle", "largeredcirle", "smallbluesquareleftoflargeyellowtriangle"]
    for w in examples:
        print(w, "->", parse_word(w))
