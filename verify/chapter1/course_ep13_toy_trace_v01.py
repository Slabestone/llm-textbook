#!/usr/bin/env python3
"""EP13: pure-Python reproducible two-position causal GPT teaching trace.

Independent educational implementation, not copied from the referenced course.
Vocabulary: A,B,C; training stream: A B C; input A B, next-token labels B C.
No network, external packages or random state.
"""
import json
import math

TOKENS = ("A", "B", "C")
EMBEDDING = {"A": (1.0, 0.0), "B": (0.0, 1.0), "C": (-1.0, 0.0)}
INPUT_IDS = (0, 1)
TARGET_IDS = (1, 2)
D = 2
X = [EMBEDDING[TOKENS[i]] for i in INPUT_IDS]


def softmax(v):
    m = max(v)
    weights = [math.exp(x - m) for x in v]
    return [x / sum(weights) for x in weights]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def trace():
    # Toy one-head attention Q=K=V=X; Wq=Wk=Wv=identity.
    raw_scores = [[dot(x, k) / math.sqrt(D) for k in X] for x in X]
    masked_scores = [
        [score if j <= i else -math.inf for j, score in enumerate(row)]
        for i, row in enumerate(raw_scores)
    ]
    alpha = [softmax(s) for s in masked_scores]
    attention = [
        [sum(alpha[i][j] * X[j][d] for j in range(2)) for d in range(D)]
        for i in range(2)
    ]
    y = [[X[i][d] + attention[i][d] for d in range(D)] for i in range(2)]
    # Tied output weights: logits for token v equal <y, e_v>.
    logits = [[dot(row, EMBEDDING[token]) for token in TOKENS] for row in y]
    probability = [softmax(row) for row in logits]
    losses = [-math.log(probability[i][TARGET_IDS[i]]) for i in range(2)]
    assert len(X) == len(alpha) == len(y) == 2
    assert alpha[0] == [1.0, 0.0], alpha[0]
    assert abs(alpha[1][0] - 0.3302384506733431) < 1e-12
    assert abs(alpha[1][1] - 0.6697615493266569) < 1e-12
    assert abs(probability[1][2] - 0.09685444503145228) < 1e-12
    assert abs(sum(probability[1]) - 1.0) < 1e-12
    assert abs(sum(losses) / 2 - 2.238738811321223) < 1e-12
    return {
        "vocabulary": TOKENS, "input_tokens": ["A", "B"],
        "target_tokens": ["B", "C"], "d_model": D, "d_k": D,
        "embeddings": X, "scaled_scores": raw_scores,
        "masked_scores": [[x if math.isfinite(x) else "-inf" for x in row] for row in masked_scores],
        "attention_weights": alpha, "attention_output": attention,
        "residual_output": y, "logits": logits,
        "probabilities": probability, "per_position_cross_entropy": losses,
        "mean_cross_entropy": sum(losses) / 2,
        "calculation_notes": "No layer norm, no MLP, identity QKV maps, tied output projection, one head, no training update. Purely a structural toy demonstration.",
    }


if __name__ == "__main__":
    print(json.dumps(trace(), ensure_ascii=False, indent=2))
