#!/usr/bin/env python3
"""Chapter 9 teaching comparison: SGD, heavy-ball momentum, Adam, AdamW.
Deterministic 2D anisotropic quadratic, NOT an LLM benchmark.
All formulas state conventions explicitly; pure Python standard library.
"""
import json
import math

START = [0.5, 0.0]
TARGET = [1.0, 2.0]
CURVATURE = [1.0, 25.0]
ETA = 0.02
STEPS = 30
BETA1, BETA2, EPS = 0.9, 0.999, 1e-8
DECAY = 0.1


def loss(w):
    return sum(0.5*c*(x-t)**2 for x,t,c in zip(w,TARGET,CURVATURE))


def gradient(w):
    return [c*(x-t) for x,t,c in zip(w,TARGET,CURVATURE)]


def fit(kind):
    w = START.copy()
    velocity = [0.0, 0.0]
    m = [0.0, 0.0]
    v = [0.0, 0.0]
    history = [{"step":0, "loss":loss(w), "w":w.copy()}]
    for t in range(1,STEPS+1):
        g = gradient(w)
        if kind == "sgd":
            update = g
        elif kind == "momentum":
            velocity = [BETA1*velocity[i]+g[i] for i in range(2)]
            update = velocity.copy()
        elif kind in ("adam", "adamw"):
            m = [BETA1*m[i]+(1-BETA1)*g[i] for i in range(2)]
            v = [BETA2*v[i]+(1-BETA2)*g[i]*g[i] for i in range(2)]
            mh = [m[i]/(1-BETA1**t) for i in range(2)]
            vh = [v[i]/(1-BETA2**t) for i in range(2)]
            update = [mh[i]/(math.sqrt(vh[i])+EPS) for i in range(2)]
        else:
            raise ValueError(kind)

        if kind == "adamw":
            # Decoupled weight decay is not fed into g, m, or v.
            w = [(1-ETA*DECAY)*w[i]-ETA*update[i] for i in range(2)]
        else:
            w = [w[i]-ETA*update[i] for i in range(2)]
        assert math.isfinite(loss(w))
        history.append({"step":t,"loss":loss(w),"w":w.copy()})
    return history


def numeric_gradient(w, i, delta=1e-5):
    wp=w.copy(); wm=w.copy()
    wp[i]+=delta; wm[i]-=delta
    return (loss(wp)-loss(wm))/(2*delta)


def main():
    error=max(abs(a-numeric_gradient(START,i)) for i,a in enumerate(gradient(START)))
    assert error < 1e-8
    data={name:fit(name) for name in ("sgd","momentum","adam","adamw")}
    assert abs(data['sgd'][1]['w'][0]-(START[0]-ETA*gradient(START)[0])) < 1e-14
    assert abs(data['momentum'][1]['w'][0]-data['sgd'][1]['w'][0])<1e-14
    assert abs((data['adam'][1]['w'][0]-data['adamw'][1]['w'][0])-ETA*DECAY*START[0])<1e-11
    assert all(len(rows)==STEPS+1 for rows in data.values())
    print(json.dumps({
        "status":"PASS","problem":"0.5*(w1-1)^2+12.5*(w2-2)^2",
        "initial":START,"eta":ETA,"steps":STEPS,"beta1":BETA1,"beta2":BETA2,
        "adam_epsilon":EPS,"adamw_weight_decay":DECAY,
        "finite_difference_gradient_error":error,
        "first_update":{name:rows[1] for name,rows in data.items()},
        "final":{name:rows[-1] for name,rows in data.items()},
        "history":data,
        "scope":"Synthetic two-dimensional quadratic; no stochastic mini-batches, no Muon benchmark",
    },ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
