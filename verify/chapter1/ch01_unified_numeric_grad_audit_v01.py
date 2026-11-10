#!/usr/bin/env python3
"""Independent, deterministic check for LLM textbook volume I (Chapters 2–5).

Educational toy only: V={A,B,C}, 2D embeddings, Q=K=V=X,
one masked attention head, residual, tied output head, no MLP/Norm,
no training updates. Does not require numpy or a network connection.
"""
import json
import math

T = ("A", "B", "C")
E = {"A": (1.0, 0.0), "B": (0.0, 1.0), "C": (-1.0, 0.0)}
X = [E["A"], E["B"]]
Y_TARGET = ("B", "C")


def dot(a, b):
    return sum(u * v for u, v in zip(a, b))


def softmax(z):
    center = max(z)
    ex = [math.exp(v - center) for v in z]
    d = sum(ex)
    return [v / d for v in ex]


def nll(z, target_index):
    return -math.log(softmax(z)[target_index])


scores = [[dot(q, k) / math.sqrt(2) for k in X] for q in X]
scores[0][1] = -math.inf  # causal: the first query cannot see the second key
A = [softmax(row) for row in scores]
H = [[sum(A[i][j] * X[j][k] for j in range(2)) for k in range(2)] for i in range(2)]
Y = [[X[i][k] + H[i][k] for k in range(2)] for i in range(2)]
Z = [[dot(y, E[t]) for t in T] for y in Y]
P = [softmax(z) for z in Z]
losses = [nll(z, T.index(target)) for z, target in zip(Z, Y_TARGET)]
mean_loss = sum(losses) / 2
analytic_grad = [P[1][j] - (1 if j == 2 else 0) for j in range(3)]

eps = 1e-5
numeric_grad = []
for j in range(3):
    high = Z[1].copy()
    low = Z[1].copy()
    high[j] += eps
    low[j] -= eps
    numeric_grad.append((nll(high, 2) - nll(low, 2)) / (2 * eps))

max_abs_gradient_error = max(abs(a - b) for a, b in zip(analytic_grad, numeric_grad))
assert A[0] == [1.0, 0.0]
assert abs(A[1][0] - 0.3302384506733431) < 1e-12
assert abs(P[1][2] - 0.09685444503145228) < 1e-12
assert abs(mean_loss - 2.238738811321223) < 1e-12
assert max_abs_gradient_error < 1e-9
assert all(abs(sum(row) - 1.0) < 1e-12 for row in P)

print(json.dumps({
    "result": "PASS",
    "vocabulary": T,
    "attention_weights_second_row": A[1],
    "vocabulary_probabilities_second_row": P[1],
    "mean_cross_entropy": mean_loss,
    "analytic_gradient": analytic_grad,
    "numerical_gradient": numeric_grad,
    "epsilon": eps,
    "max_abs_gradient_error": max_abs_gradient_error,
}, indent=2, ensure_ascii=False))
