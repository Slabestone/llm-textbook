#!/usr/bin/env python3
"""EP28: task-aware compression versus reconstruction distortion.
Independent 1D finite-level scalar quantization toy. NOT VQ-VAE/FSQ training
and not a robot/world-model benchmark; it illustrates why MSE and control
decision accuracy need not rank representations the same way.
"""
import json
import math
from collections import Counter

VALUES = (-0.9,-0.3,-0.1,0.1,0.3,0.9)

def quantize_scalar(value, levels):
    # Unique nearest level for our sample values; ties deterministically choose
    # smallest absolute level, then smaller numeric level.
    return min(levels, key=lambda u: (abs(value-u),abs(u),u))

def decision(v):
    # Action threshold: 0 belongs to the positive class by explicit convention.
    return "right" if v >= 0 else "left"

def entropy(codes):
    count=Counter(codes)
    n=len(codes)
    return -sum((v/n)*math.log2(v/n) for v in count.values())

def scheme(levels):
    recon=[quantize_scalar(v,levels) for v in VALUES]
    err=[(v-q)**2 for v,q in zip(VALUES,recon)]
    mistakes=sum(decision(v)!=decision(q) for v,q in zip(VALUES,recon))
    k=len(levels)
    return {
        "levels":levels,"codebook_size":k,
        "minimum_fixed_binary_code_bits_per_sample":math.ceil(math.log2(k)),
        "ideal_code_length_if_uniform":math.log2(k),
        "empirical_source_code_entropy_bits":entropy(recon),
        "reconstruction":recon,"squared_errors":err,
        "mse":sum(err)/len(err),
        "decision_errors":mistakes,
        "decision_accuracy":1-mistakes/len(VALUES),
    }

def main():
    high_resolution=scheme((-1.0,-0.5,0.0,0.5,1.0))
    sign_preserving=scheme((-0.5,0.5))
    assert abs(high_resolution["mse"]-0.02)<1e-12
    assert abs(sign_preserving["mse"]-0.12)<1e-12
    assert high_resolution["decision_errors"]==1
    assert sign_preserving["decision_errors"]==0
    assert high_resolution["minimum_fixed_binary_code_bits_per_sample"]==3
    assert sign_preserving["minimum_fixed_binary_code_bits_per_sample"]==1
    assert high_resolution["mse"]<sign_preserving["mse"]
    assert high_resolution["decision_accuracy"]<sign_preserving["decision_accuracy"]
    print(json.dumps({
        "input_continuous_action_values":VALUES,
        "task":"binary left/right decision from sign; zero reconstructed values classified RIGHT",
        "fine_uniform_scalar_grid":high_resolution,
        "coarse_sign_preserving_grid":sign_preserving,
        "ranking":"5-level has lower MSE, 2-level has higher sign-action accuracy and lower fixed-length rate",
        "science_boundary":[
            "finite-level scalar quantizers only; not an implementation of learned VQ-VAE or a trained FSQ codec",
            "5-level uniform grid and 2-level sign grid are toy fixed quantizers; FSQ proper uses projected bounded scalar channels with implicit product codebook",
            "task accuracy depends on chosen threshold and input distribution; the finding is an existence counterexample not a universal ranking",
            "fixed code length ceil(log2(K)) is distinct from log2(K) and empirical entropy; real rate depends on coding overhead",
            "continuous representation with finite-precision numbers has finite bits; do not claim literally infinite storage",
            "improving reconstruction MSE does not mathematically guarantee downstream task success"
        ],
        "sources":{
            "vq_vae":"https://arxiv.org/abs/1711.00937",
            "fsq":"https://arxiv.org/abs/2309.15505"
        }
    },ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
