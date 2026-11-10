#!/usr/bin/env python3
"""EP30 deterministic block-causal visibility and mask training teaching toy.

Eight ordered tokens A..H; block size 1/2/4/8.
Mask M[i,j]=1 if j's block is at or before i's block; within the
same block positions can attend in BOTH directions. This is a pedagogical
architecture pattern, not a reproduction of a particular BD3-LM or video
diffusion network, its noise scheduler or doubling trick.
"""
import json

T=8
TOKENS=list("ABCDEFGH")

def mask(block_size):
    assert T%block_size==0
    return [[int(j//block_size<=i//block_size) for j in range(T)] for i in range(T)]

def prepare_active_block(block_size=2,active_block=1):
    assert block_size==2 and active_block==1
    start=active_block*block_size
    end=start+block_size
    visible=[TOKENS[i] if i<start else "[MASK]" for i in range(T)]
    # Mask *all* current tokens, and all future tokens, to avoid accidental
    # target copying or future contamination in this extreme toy.
    loss=[int(start<=i<end) for i in range(T)]
    allowed=mask(block_size)
    assert visible==["A","B","[MASK]","[MASK]","[MASK]","[MASK]","[MASK]","[MASK]"]
    assert loss==[0,0,1,1,0,0,0,0]
    assert allowed[2][3]==1 and allowed[3][2]==1
    assert all(allowed[i][j]==0 for i in (2,3) for j in range(4,8))
    # A very bad training construction that exposes the clean target token
    # for the active block; a network can trivially copy the answer.
    bad=TOKENS[:]
    assert bad[2:4]==TOKENS[2:4] and visible[2:4]!=TOKENS[2:4]
    return {"input_tokens_masked":visible,
            "loss_positions":loss,
            "target_labels":TOKENS[start:end],
            "bad_clean_teacher_forced_active_block":bad,
            "note":"This is an all-masked current-block demonstration. Real discrete diffusion uses sampled noise levels / partial masking and a chosen loss weighting."}

def main():
    variants={}
    counts={}
    for b in (1,2,4,8):
        m=mask(b)
        total=sum(sum(row) for row in m)
        formula=sum(b*b*(r+1) for r in range(T//b))
        assert total==formula
        assert all(m[i][j]==(1 if j//b<=i//b else 0) for i in range(T) for j in range(T))
        variants[str(b)]={"block_size":b,"n_blocks":T//b,"visible_attention_pairs":total,"mask":m}
        counts[b]=total
    assert counts=={1:36,2:40,4:48,8:64}
    # Forward passes counted assuming S serial denoising calls per block.
    S=4
    calls={str(b):(T//b)*S for b in (1,2,4,8)}
    assert calls=={"1":32,"2":16,"4":8,"8":4}
    print(json.dumps({
       "schema":"EP30_block_mask_toy_v01",
       "tokens":TOKENS,"T":T,
       "block_masks":variants,
       "mask_block_2_active_second_block":prepare_active_block(),
       "illustrative_sampler_calls":{
          "assumed_serial_denoising_steps_per_block":S,
          "calls_per_block_size":calls,
          "warning":"call counts do NOT measure FLOPs, latency, output quality or true autoregressive equivalence; per-call compute and caching differ"},
       "factorization":"p(x1..x8)=product_b p(x_block_b | previous clean blocks). Diffusion may model each conditional block joint.",
       "scope":"purely structural visibility and scheduling, no diffusion model trained or sampled",
       "papers":["https://arxiv.org/abs/2503.09573","https://arxiv.org/abs/2506.08009"]
    },ensure_ascii=False,indent=2))
if __name__=="__main__":
    main()
