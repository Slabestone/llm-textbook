# llm-textbook

Public verification scripts and **unified bibliography** for the Chinese textbook  
《从语言模型到大语言模型》(From Language Models to Large Language Models).

Editor / scientific lead: **谢承融**.  
This repository is the **public** place to recompute textbook toy examples and run PDF structural gates — **no Mac-local paths required**.

Textbook manuscript sources remain on the author’s machine / private archives.  
Canonical public URL: <https://github.com/Slabestone/llm-textbook>

## Layout

| Path | Contents |
|---|---|
| `verify/chapter1/` | Bigram / overfitting / optimizer / mini-Transformer / EP13 Attention toy / gradient audit |
| `verify/chapter2/` | Volume-2 arithmetic checks |
| `verify/chapter4/` | SFT/LoRA/DPO/Inference/RAG/Agent quantitative figure generator |
| `verify/chapter5/` | EP28–32 multimodal toy scripts cited by Chapter 5 |
| `data/` | `.dat` / `.json` curve tables used by book figures |
| `references/bibliography.tex` | Unified Author `(year)` bibliography block |

## Quick start

```bash
git clone https://github.com/Slabestone/llm-textbook.git
cd llm-textbook

python3 verify/chapter2/volume2_checks.py
python3 verify/chapter1/llm_vol1_ch10_bigram_train_eval_v01.py
python3 verify/chapter1/course_ep13_toy_trace_v01.py
python3 verify/chapter1/ch01_unified_numeric_grad_audit_v01.py
python3 verify/chapter4/llm_iv_v05_plot_recompute.py
```

Chapter 5 toys (examples):

```bash
python3 verify/chapter5/ep28_vq_fsq_nearest_code_math_toy_v01.py
python3 verify/chapter5/ep29_diffusion_flow_matching_toy_v01.py
```

## Figure 1.17: early stopping (fully reproducible)

The training/validation curves in §1.11.5 come from a deterministic,
16-parameter conditional Bigram experiment (not Transformer training).
Regenerate all 121 data rows and verify them **byte-for-byte** against
the public plotting table:

```bash
python3 verify/chapter1/reproduce_figure_1_17.py
```

See [Figure 1.17 provenance, selected checkpoint vs. raw minimum,
and plotting instructions](verify/chapter1/README_figure_1_17.md).
The repository also includes an independent [GitHub Actions check](.github/workflows/verify-figure-1-17.yml).

## Name map (book prose → public path)

| Book historical filename | Public path |
|---|---|
| `LLM_单书_第1章_微型Transformer复算_v0.3_….py` | `verify/chapter1/ch01_mini_transformer_recompute_v03.py` |
| `llm_vol1_ch10_bigram_train_generate_v02.py` | `verify/chapter1/llm_vol1_ch10_bigram_train_generate_v02.py` |
| `05_制作工程/course_ep13_toy_trace_v01.py` | `verify/chapter1/course_ep13_toy_trace_v01.py` |
| `LLM_教材_第一卷_统一数值与梯度复核_v0.1_….py` | `verify/chapter1/ch01_unified_numeric_grad_audit_v01.py` |
| `LLM_IV_v0.5_定量作图与复算.py` | `verify/chapter4/llm_iv_v05_plot_recompute.py` |
| `LLM_IV_v0.5_计算图原始数据.json` | `data/llm_iv_v05_figure_raw_data.json` |
| `05_制作工程/ep28_….py` … `ep32_….py` | `verify/chapter5/ep28_….py` … |

Cite this repository from the book as key `llmtextbook`.

## Version

Aligned with textbook **v2.8** (2026-11-11): public path replacement pass for remaining Mac-local citations.

## License

Code: MIT.  
Bibliography metadata: CC0.  
Textbook prose/PDF are **not** relicensed by this repository.
