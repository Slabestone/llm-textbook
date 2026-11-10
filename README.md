# llm-textbook

Public verification scripts and **unified bibliography** for the Chinese textbook  
《从语言模型到大语言模型》(From Language Models to Large Language Models).

Editor / scientific lead: **谢承融**.  
This repository is the **public** place to recompute textbook toy examples and run PDF structural gates — **no Mac-local paths required**.

Textbook manuscript sources remain on the author’s machine / private archives.  
Canonical public URL: <https://github.com/Slabestone/llm-textbook>

## Quick start

```bash
git clone https://github.com/Slabestone/llm-textbook.git
cd llm-textbook

# Chapter 2 arithmetic (stdlib only)
python3 verify/chapter2/volume2_checks.py
python3 verify/chapter2/volume2_extended_checks.py

# Chapter 1 toy experiments (stdlib / numpy if listed in script headers)
python3 verify/chapter1/llm_vol1_ch08_overfitting_v04.py
python3 verify/chapter1/llm_vol1_ch09_optimizer_2d_v01.py
python3 verify/chapter1/llm_vol1_ch10_bigram_train_eval_v01.py
```

Curve `.dat` files used by the book figures live under `data/`.

PDF structural gate (optional; needs `pypdf`):

```bash
python3 -m pip install pypdf
python3 verify/preflight/pdf_structure_gate_v02.py /path/to/book.pdf /path/to/book.log
```

## Bibliography

Unified Author `(year). Title. Venue. URL.` entries:

- [`references/bibliography.tex`](references/bibliography.tex) — `\bibitem{...}` block for XeLaTeX `\begin{thebibliography}`

Cite this repository from the book as key `llmtextbook`.

## Version

Aligned with textbook **v2.6** (2026-11-11).

## License

Code: MIT.  
Bibliography metadata: CC0.  
Textbook prose/PDF are **not** relicensed by this repository.
