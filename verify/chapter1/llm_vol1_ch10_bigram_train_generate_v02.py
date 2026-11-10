#!/usr/bin/env python3
"""Vol I chapter 10: deterministic next-token training, validation, and generation loop.

Four artificial tokens A/B/C/D. Model logits W[current_token, next_token].
True generative P(next|current) is a fixed 4x4 conditional table.
Train/validation/test pairs are independently drawn (same conditional
distribution, distinct random draws; identical token-pair VALUES can recur).
Analytic full-batch CE gradient, finite differences, SGD, early stopping,
one held-out test evaluation. No external libraries or network. This is
a BIGRAM next-token LM, NOT Transformer or full GPT pretraining.
"""
import json
import math
import random
from copy import deepcopy

TOKENS = "ABCD"
V = len(TOKENS)
P_TRUE = [
    [0.04, 0.78, 0.12, 0.06],
    [0.05, 0.08, 0.81, 0.06],
    [0.06, 0.08, 0.08, 0.78],
    [0.78, 0.08, 0.08, 0.06],
]
N_TRAIN, N_VAL, N_TEST = 32, 400, 400
SEEDS = {"train": 19, "validation": 23, "test": 29}
LR, STEPS, PATIENCE, MIN_DELTA = 1.0, 120, 12, 1e-4


def make_pairs(n, seed):
    rng = random.Random(seed)
    pairs = []
    for sample_index in range(n):
        ctx = sample_index % V
        u = rng.random()
        running = 0.0
        choice = V - 1
        for target, pr in enumerate(P_TRUE[ctx]):
            running += pr
            if u < running:
                choice = target
                break
        pairs.append((ctx, choice))
    rng.shuffle(pairs)
    return pairs


def softmax(x):
    pivot = max(x)
    ex = [math.exp(y - pivot) for y in x]
    z = sum(ex)
    return [a / z for a in ex]


def loss(w, pairs):
    n = len(pairs)
    return -sum(math.log(softmax(w[c])[y]) for c, y in pairs) / n


def grad(w, pairs):
    g = [[0.0] * V for _ in range(V)]
    for c, y in pairs:
        probabilities = softmax(w[c])
        for j in range(V):
            g[c][j] += (probabilities[j] - (1.0 if j == y else 0.0)) / len(pairs)
    return g


def gradient_fd(w, pairs, r, s, delta=1e-5):
    wp, wm = deepcopy(w), deepcopy(w)
    wp[r][s] += delta
    wm[r][s] -= delta
    return (loss(wp, pairs) - loss(wm, pairs)) / (2 * delta)


def update(w, g, eta):
    return [[w[i][j] - eta * g[i][j] for j in range(V)] for i in range(V)]


def p_choose(w, token):
    return dict(zip(TOKENS, softmax(w[TOKENS.index(token)])))



def generate(w, prompt, n_new, *, temperature=1.0, sample=False, seed=0):
    """Autoregressively extend a token-string prompt using fixed trained logits."""
    if not prompt or any(token not in TOKENS for token in prompt):
        raise ValueError("prompt must contain one or more tokens from TOKENS")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    rng = random.Random(seed)
    sequence = list(prompt)
    for _ in range(n_new):
        logits = w[TOKENS.index(sequence[-1])]
        probabilities = softmax([z / temperature for z in logits])
        if sample:
            u = rng.random()
            cumulative = 0.0
            next_index = V - 1
            for j, probability in enumerate(probabilities):
                cumulative += probability
                if u < cumulative:
                    next_index = j
                    break
        else:
            next_index = max(range(V), key=probabilities.__getitem__)
        sequence.append(TOKENS[next_index])
    return "".join(sequence)


def main():
    tr = make_pairs(N_TRAIN, SEEDS["train"])
    val = make_pairs(N_VAL, SEEDS["validation"])
    test = make_pairs(N_TEST, SEEDS["test"])
    assert len(tr) == N_TRAIN and len(val) == N_VAL and len(test) == N_TEST
    assert all(abs(sum(row) - 1) < 1e-12 for row in P_TRUE)

    w = [[0.0] * V for _ in range(V)]
    g = grad(w, tr)
    max_fd_error = max(abs(g[i][j] - gradient_fd(w, tr, i, j))
                       for i in range(V) for j in range(V))
    assert max_fd_error < 1e-9

    rows, weights_by_step = [], []
    best_val, best_step, best_w = math.inf, -1, None
    last_improvement_step = 0
    stop_step = None

    # Run fixed-budget reference trajectory. Early-stop would return the
    # best checkpoint at stop_step; continuation to STEPS is for comparison.
    for step in range(STEPS + 1):
        ltr = loss(w, tr)
        lval = loss(w, val)
        rows.append({"step": step, "train": ltr, "validation": lval})
        weights_by_step.append(deepcopy(w))
        if stop_step is None:
            if lval < best_val - MIN_DELTA:
                best_val = lval
                best_step = step
                best_w = deepcopy(w)
                last_improvement_step = step
            elif step - last_improvement_step >= PATIENCE:
                stop_step = step
        if step != STEPS:
            w = update(w, grad(w, tr), LR)

    assert best_step >= 1
    assert stop_step is not None and stop_step > best_step
    assert loss(weights_by_step[STEPS], tr) < loss(best_w, tr)
    assert loss(weights_by_step[STEPS], val) > loss(best_w, val)

    # Evaluate test only after the stopping decision has been made.
    # Both evaluations are for *description*, not checkpoint selection.
    test_at_selected = loss(best_w, test)
    test_at_full = loss(weights_by_step[STEPS], test)
    summary = {
        "status": "PASS",
        "model": "4-token conditional bigram LM, logits W[context,next], 16 parameters",
        "data_generator": "independent conditional samples from a fixed 4x4 transition table",
        "shapes": {"W": [V, V]},
        "n": {"train": N_TRAIN, "validation": N_VAL, "test": N_TEST},
        "seeds": SEEDS,
        "learning_rate": LR, "reference_steps": STEPS,
        "early_stopping": {"patience": PATIENCE, "min_delta": MIN_DELTA,
                           "stop_step": stop_step, "selected_step": best_step},
        "fd_gradient_max_error": max_fd_error,
        "initial": {"train_loss": rows[0]["train"], "validation_loss": rows[0]["validation"]},
        "selected": {"epoch": best_step, "train_loss": loss(best_w, tr),
                     "validation_loss": loss(best_w, val),
                     "test_loss": test_at_selected, "test_ppl": math.exp(test_at_selected),
                     "p_next_given_A": p_choose(best_w, "A")},
        "overtrained": {"epoch": STEPS, "train_loss": rows[-1]["train"],
                        "validation_loss": rows[-1]["validation"],
                        "test_loss": test_at_full, "test_ppl": math.exp(test_at_full)},
        "true_conditional_p_next_given_A": dict(zip(TOKENS, P_TRUE[0])),
        "observed_training_counts": {
            TOKENS[c]: {TOKENS[y]: sum(1 for ci, yi in tr if ci == c and yi == y) for y in range(V)}
            for c in range(V)
        },
        "series": rows,
        "limitations": [
            "Token pairs are independently sampled and can have identical values across splits",
            "This is not a Transformer; only a 4x4 transition-logit table is trained",
            "This one seed is an illustrative case, not a robustness or model-ranking study",
            "Test is not used in the gradient or early-stopping rule",
            "The fixed-step run continuing after the stopping point is a counterfactual demonstration",
            "An early-stopping result is conditional on the chosen validation distribution",
        ],
    }
    summary["generation"] = {
        "prompt": "A",
        "new_tokens": 12,
        "temperature_for_sampling": 1.5,
        "greedy": generate(best_w, "A", 12),
        "sampled": generate(best_w, "A", 12, temperature=1.5,
                            sample=True, seed=17),
        "parameters_updated_during_generation": False,
        "stopping_rule": "fixed 12 new tokens; vocabulary has no EOS token",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
