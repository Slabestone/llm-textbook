#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第四卷定量插图：可复算计算与可选 Matplotlib 绘图。
运行:
    python3 LLM_IV_v0.5_定量作图与复算.py --check
    python3 LLM_IV_v0.5_定量作图与复算.py --render --output-dir ./figures
所有图均为教学构造数据。默认只打印 JSON，不读写文件。
"""
import argparse
import json
import math
from pathlib import Path

def logit(p):
    return math.log(p/(1-p))

def softplus(x):
    return math.log1p(math.exp(-abs(x)))+max(x,0.0)

def dpo_loss(p,q,beta=2.0):
    # L=-log sigma(beta(logit(p)-logit(q)))
    return softplus(-beta*(logit(p)-logit(q)))

def lora_truncation_error(rank, singular_values):
    top=sum(s*s for s in singular_values[:rank])
    total=sum(s*s for s in singular_values)
    return math.sqrt(max(0.0,1-top/total))

def infer_coverage(n,p):
    return 1.0-(1.0-p)**n

def verifier_selected_correct(n,p,s,f):
    # 独立候选；逐个接受；若有多个被接受，从中等概率选一个
    a=p*s+(1-p)*f
    ppv=p*s/a
    return ppv*(1-(1-a)**n)

def get_data():
    ranks=[1,2,4,8,16,32,64]
    # 合成 rank-64 矩阵的谱：sigma_i=(0.92)^i，展示截断近似误差
    sigma=[.92**i for i in range(64)]
    items={
      "sft":[{"p":round(i/20,2),"nll":-math.log(i/20)} for i in range(1,20)],
      "lora":[{"r":r,"err":lora_truncation_error(r,sigma),
               "trainable_ratio":(8192*r)/(4096*4096)} for r in ranks],
      "dpo":[{"p":round(i/20,2),"q05":dpo_loss(i/20,.5),
              "q08":dpo_loss(i/20,.8)} for i in range(2,19)],
      "inference":[{"n":n,"oracle":infer_coverage(n,.25),
                    "verified":verifier_selected_correct(n,.25,.9,.1)} for n in range(1,21)],
      "rag":[{"k":k,"recall":sum(x in {"v2","rule"} for x in ["v1","v2","rule","decoy"][:k])/2,
              "precision":sum(x in {"v2","rule"} for x in ["v1","v2","rule","decoy"][:k])/k} for k in range(1,5)],
      "agent":[{"attempts":n,"new_keys":n,"same_key":1} for n in range(1,7)]
    }
    return items

def check(d):
    assert abs(d["sft"][9]["nll"]-math.log(2))<1e-12
    assert d["lora"][-1]["err"] == 0.0
    assert abs(d["lora"][3]["trainable_ratio"]-16/4096)<1e-12
    assert abs(dpo_loss(.6,.8,2)-2.093234863812172)<1e-10
    assert abs(d["inference"][3]["oracle"]-.68359375)<1e-12
    assert abs(d["inference"][3]["verified"]-.75*(1-.7**4))<1e-12
    assert d["rag"][2]["recall"]==1 and abs(d["rag"][2]["precision"]-2/3)<1e-12
    assert all(x["same_key"]==1 for x in d["agent"])
    for name in d:
        assert all(all(math.isfinite(v) for v in row.values()) for row in d[name])
    return True

def render(d,output_dir):
    # 可选运行；不依赖 seaborn。不设置 matplotlib 全局颜色主题。
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    output_dir.mkdir(parents=True,exist_ok=True)
    configs=[
      ("sft","p(correct token)","negative log-likelihood",
       [("nll","-log(p)")],"p"),
      ("lora","rank r","fraction",
       [("err","relative SVD truncation error"),("trainable_ratio","LoRA trainable parameter fraction")],"r"),
      ("dpo","preferred policy p","DPO loss",
       [("q05","reference q=0.5"),("q08","reference q=0.8")],"p"),
      ("inference","number of samples N","probability",
       [("oracle","at least one correct candidate"),("verified","random accepted candidate correct")],"n"),
      ("rag","retrieved top-k","fraction",
       [("recall","Recall@k"),("precision","Precision@k")],"k"),
      ("agent","dispatch attempts","committed side effects",
       [("new_keys","new key on every retry"),("same_key","stable idempotency key")],"attempts")
    ]
    for name,xlabel,ylabel,series,xkey in configs:
        fig,ax=plt.subplots(figsize=(7.2,4.2))
        for key,label in series:
            ax.plot([row[xkey] for row in d[name]],[row[key] for row in d[name]],
                    marker="o" if name in ("rag","agent","lora") else None,label=label,linewidth=2)
        ax.set(xlabel=xlabel,ylabel=ylabel)
        ax.grid(True,alpha=.2)
        ax.legend(frameon=False)
        if name in ("rag","agent"):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        if name in ("sft","inference","rag"):
            ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(output_dir/(name+".png"),dpi=180,bbox_inches="tight")
        fig.savefig(output_dir/(name+".pdf"),bbox_inches="tight")
        plt.close(fig)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--check",action="store_true")
    parser.add_argument("--render",action="store_true")
    parser.add_argument("--output-dir",type=Path,default=Path("./figure_outputs"))
    args=parser.parse_args()
    d=get_data()
    check(d)
    if args.render:
        render(d,args.output_dir)
        print("Rendered 6 Matplotlib PNG/PDF pairs to", args.output_dir)
    else:
        print(json.dumps(d,ensure_ascii=False,separators=(",",":")))

if __name__=="__main__":
    main()
