#!/usr/bin/env python3
"""Chapter 8: controlled train/validation overfitting example.

Synthetic target y=sin(pi*x)+label_noise, only standard library.
9 equispaced training points; 200 disjoint validation points.
Finite-degree polynomial regression solved via normal equations with
partial-pivoting Gauss-Jordan; a teaching demonstration, not an LLM fit.
The fixed RNG seed is part of the definition, not a claim of robustness.
"""
import json
import math
import random

SEED = 17
NOISE = 0.08
NTRAIN = 9
NVALID = 200
DEGREES = (1, 3, 8)

def solve(mat, rhs):
    n = len(rhs)
    a = [[float(z) for z in row] + [float(rhs[i])] for i, row in enumerate(mat)]
    for j in range(n):
        pivot = max(range(j, n), key=lambda i: abs(a[i][j]))
        a[j], a[pivot] = a[pivot], a[j]
        d = a[j][j]
        if abs(d) < 1e-12:
            raise ArithmeticError("Nearly singular fit; do not silently trust coefficients")
        for k in range(j, n + 1):
            a[j][k] /= d
        for i in range(n):
            if i == j: continue
            mul = a[i][j]
            for k in range(j, n + 1):
                a[i][k] -= mul * a[j][k]
    return [row[-1] for row in a]

def fit(xs, ys, degree):
    normal = [[sum(x ** (i + j) for x in xs) for j in range(degree + 1)]
              for i in range(degree + 1)]
    b = [sum(y * x ** j for x, y in zip(xs, ys)) for j in range(degree + 1)]
    return solve(normal, b)

def evaluate(coeff, x):
    # Horner's method for stable and efficient evaluation
    out = 0.0
    for value in reversed(coeff):
        out = out * x + value
    return out

def mse(w, xs, ys):
    return sum((evaluate(w, x) - y) ** 2 for x, y in zip(xs, ys)) / len(xs)

def main():
    rng = random.Random(SEED)
    xtrain = [-1 + 2 * i / (NTRAIN - 1) for i in range(NTRAIN)]
    ytrain_clean = [math.sin(math.pi * x) for x in xtrain]
    ytrain = [y + NOISE * rng.choice((-1, 1)) for y in ytrain_clean]
    xvalid = [-1 + 2 * (i + 0.5) / NVALID for i in range(NVALID)]
    yvalid_clean = [math.sin(math.pi * x) for x in xvalid]
    # Validation labels have independently generated measurement noise.
    yvalid = [y + NOISE * rng.choice((-1, 1)) for y in yvalid_clean]

    results = {}
    weights = {}
    for degree in DEGREES:
        w = fit(xtrain, ytrain, degree)
        weights[degree] = w
        results[str(degree)] = {
            "train_mse_noisy_targets": mse(w, xtrain, ytrain),
            "validation_mse_independent_noisy_targets": mse(w, xvalid, yvalid),
            "validation_mse_clean_function": mse(w, xvalid, yvalid_clean),
            "number_parameters": degree + 1,
            "max_abs_weight": max(abs(t) for t in w),
        }
    assert results['8']['train_mse_noisy_targets'] < 1e-9
    assert results['3']['train_mse_noisy_targets'] > results['8']['train_mse_noisy_targets']
    assert results['8']['validation_mse_independent_noisy_targets'] > 2 * results['3']['validation_mse_independent_noisy_targets']
    # Intended example: validation/test do not overlap train by construction.
    assert all(min(abs(t - v) for t in xtrain) > 1e-6 for v in xvalid)
    # Readable plotting data for pgfplots, with actual calculated predictions.
    curves = [{"x":x, "true":math.sin(math.pi*x), "degree3":evaluate(weights[3],x),
               "degree8":evaluate(weights[8],x)} for x in [-1 + i / 100 for i in range(201)]]
    print(json.dumps({
        "status":"PASS", "seed":SEED,"noise_amplitude":NOISE,"train_points":NTRAIN,
        "validation_points":NVALID,"measured":results,
        "train":list(zip(xtrain,ytrain)), "curves":curves,
        "caveats":["Only one seeded synthetic realization",
                   "Polynomial regression, NOT a Transformer",
                   "Training labels and validation labels have independent noise",
                   "Validation was not used in parameter fitting",
                   "Choosing degrees based on this validation introduces model-selection bias"],
    },ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
