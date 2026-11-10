#!/usr/bin/env python3
"""EP28 vector quantization and finite scalar quantization (FSQ) arithmetic toy.
No encoder/decoder is trained. Illustrates nearest-code selection, Cartesian
product codebook, fixed binary index width and reconstruction distortion.
"""
import json, math
z=(0.7,0.2)
VQ=[(0.,0.),(1.,0.),(0.,1.)]
dists=[sum((a-b)**2 for a,b in zip(z,e)) for e in VQ]
best=min(range(len(VQ)),key=lambda j:dists[j])
assert best==1
assert all(abs(a-b)<1e-12 for a,b in zip(dists,[.53,.13,1.13]))
LEVELS=[(-1.,-.5,0.,.5,1.),(-1.,-.25,.25,1.)]
inds=[min(range(len(L)),key=lambda i:(abs(z[j]-L[i]),i)) for j,L in enumerate(LEVELS)]
q=tuple(LEVELS[j][i] for j,i in enumerate(inds))
code_idx=inds[0]*len(LEVELS[1])+inds[1]
K=math.prod(map(len,LEVELS))
bits=math.ceil(math.log2(K))
mse=sum((a-b)**2 for a,b in zip(z,q))/len(z)
assert q==(.5,.25) and code_idx==14 and K==20 and bits==5
print(json.dumps({"scope":"arithmetic toy only; not VQ-VAE or trained FSQ",
                  "latent":z,
                  "VQ":{"codebook":VQ,"squared_distances":dists,"nearest_code_index":best,"nearest_vector":VQ[best]},
                  "FSQ":{"levels_by_dimension":LEVELS,"index_per_dimension":inds,"quantized_vector":q,
                    "implicit_codebook_size":K,"flat_index_zero_based":code_idx,
                    "fixed_length_bits":bits,"log2K_ideal_noninteger":math.log2(K),
                    "MSE_per_component":mse},
                  "warning":"VQ and FSQ codebook use, bits/element and actual compression require distribution, entropy coding, projection, packing and decoder costs."},ensure_ascii=False,indent=2))
