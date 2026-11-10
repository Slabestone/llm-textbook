#!/usr/bin/env python3
"""EP31: 3D axial RoPE and hypothetical real-time scheduling exact math.

Not official VideoRoPE implementation; no trained video model and no GPU timing.
A 6-dimensional toy query/key has one 2D rotary pair per axis t,y,x.
"""
import json, math

OMEGAS=(math.pi/3,math.pi/2,math.pi/4)
SHAPE=(2,4,4)

def rotate_pair(theta):
    return (math.cos(theta),math.sin(theta))

def embedding(coords):
    return [u for axis,angle in zip(coords,OMEGAS)
              for u in rotate_pair(axis*angle)]

def dot(a,b): return sum(x*y for x,y in zip(a,b))
def l2(a):return math.sqrt(sum(x*x for x in a))

def rope():
    a=(1,2,3); b=(2,1,3)
    va,vb=embedding(a),embedding(b)
    result=dot(va,vb)
    expected=sum(math.cos(angle*(x-y)) for x,y,angle in zip(a,b,OMEGAS))
    shifted=(tuple(v+s for v,s in zip(a,(4,10,20))),
             tuple(v+s for v,s in zip(b,(4,10,20))))
    assert abs(result-expected)<1e-12
    assert abs(result-1.5)<1e-12
    assert abs(dot(embedding(shifted[0]),embedding(shifted[1]))-result)<1e-12
    assert abs(l2(va)-math.sqrt(3))<1e-12
    # A flattened video index can make very different (t,y,x) coordinates
    # adjacent. Illustrates why the factorization/axis allocation matters.
    flatten=lambda p:(p[0]*SHAPE[1]+p[1])*SHAPE[2]+p[2]
    before=(0,3,3); after=(1,0,0)
    assert flatten(after)-flatten(before)==1
    assert tuple(after[i]-before[i] for i in range(3))==(1,-3,-3)
    return {"angles_rad_per_coordinate":OMEGAS,"pos_a":a,"pos_b":b,
            "query_rotated":va,"key_rotated":vb,"dot":result,
            "relative_cosine_sum":expected,"shifted_coords":shifted,
            "dot_after_shared_translation":dot(embedding(shifted[0]),embedding(shifted[1])),
            "flatten_counterexample":{"grid_shape":SHAPE,"a":before,
                "b":after,"flatten_index_a":flatten(before),"flatten_index_b":flatten(after),
                "spatiotemporal_delta":[after[i]-before[i] for i in range(3)]},
            "note":"axial 3D block-diagonal rotation toy, not a specific model config. Frequencies and axes are arbitrary educational settings."}

def realtime():
    fps=24
    chunk_frames=4
    budget=1000*chunk_frames/fps
    examples=[]
    for label,step_ms,network_ms in [("A",30,20),("B",40,20),("C",30,50)]:
        denoise_calls=4
        codec_ms=15
        total=denoise_calls*step_ms+codec_ms+network_ms
        examples.append({"case":label,"denoise_calls":denoise_calls,
          "ms_per_call":step_ms,"codec_ms":codec_ms,"network_ms":network_ms,
          "total_ms":total,"meets_chunk_deadline":total<=budget,
          "latency_margin_ms":budget-total})
    assert examples[0]["total_ms"]==155
    assert examples[1]["total_ms"]==195
    assert examples[2]["total_ms"]==185
    return {"fps":fps,"chunk_frames":chunk_frames,"frame_period_ms":1000/fps,
            "chunk_deadline_ms":budget,"cases":examples,
            "note":"toy sequential critical path; no GPU/network measurement, no overlapped execution or buffering modeled"}

if __name__=="__main__":
    print(json.dumps({"schema":"EP31_math_toy_v01","RoPE":rope(),"realtime":realtime(),
                      "science_scope":"relative phase and synthetic frame deadline arithmetic; no actual decoder, model or hardware benchmark"},
                      ensure_ascii=False,indent=2))
