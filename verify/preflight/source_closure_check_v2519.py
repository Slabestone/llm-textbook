#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible read-only verification for v2.5.19.
No subprocess, network, writes, PDF build or trusted state mutation.
Outputs JSON; exit 0 only if all pinned source and chapter invariants pass.
"""
from pathlib import Path
import hashlib, re, collections, difflib, json, sys
ROOT=Path(__file__).resolve().parent
A="LLM_单书五章_统一母稿_全书视觉排版_v2.5.18_20261110_雾岚.tex"
B="LLM_单书五章_统一母稿_节数修订_v2.5.19_20261110_雾岚.tex"
EXPECTED_A="818812554931ff3ca21dee90f79d7ed097797d1bcae5ec8f02236d97f042bf14"
EXPECTED_B="bec797f8915523e056872a5b7062724efac25137b27d52f905b05bfa8bc23f0d"
def text(n):return (ROOT/n).read_text(encoding="utf-8")
def sha(n):return hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
def clean(s):return "\n".join(re.split(r"(?<!\\)%",x,1)[0] for x in s.splitlines())
def inspect(n):
 p=(ROOT/n).resolve()
 checked={};labels=[];cites=[];bibs=[];missing=[];edges=[]
 def visit(f):
  if f in checked:return
  raw=f.read_text(encoding="utf-8");t=clean(raw)
  checked[f]=len(re.findall(r"\\section(?:\[[^\]]*\])?\s*\{",t))
  labels.extend(re.findall(r"\\label\s*\{([^}]*)\}",t))
  bibs.extend(re.findall(r"\\bibitem(?:\[[^\]]*\])?\s*\{([^}]*)\}",t))
  for keys in re.findall(r"\\cite(?:\[[^\]]*\])?\s*\{([^}]*)\}",t):
   cites.extend(y.strip() for y in keys.split(","))
  for name in re.findall(r"\\(?:input|include)\s*\{([^}]+)\}",t):
   q=f.parent/name
   if not q.suffix:q=q.with_suffix(".tex")
   edges.append((f.name,q.name))
   if q.is_file():visit(q.resolve())
   else:missing.append(str(q))
 visit(p)
 # Frontmatter and historical introduction sections do not belong to the
 # 5 chapter numbered-section total.
 excluded=sum(n for f,n in checked.items() if f==p or "全书历史导论" in f.name)
 counted=sum(checked.values())
 duplicate=[x for x,num in collections.Counter(labels).items() if num>1]
 return {
  "unique_tex":len(checked),"input_edges":len(edges),
  "missing_tex":missing,"all_section_tokens":counted,
  "frontmatter_and_history_sections":excluded,
  "numbered_five_chapter_sections":counted-excluded,
  "duplicate_labels":duplicate,
  "unresolved_bibliography_keys":sorted(set(cites)-set(bibs))
 }
old=text(A).splitlines();new=text(B).splitlines()
diff=[{"kind":kind,"old":old[i:j],"new":new[k:l]} for kind,i,j,k,l in
      difflib.SequenceMatcher(a=old,b=new,autojunk=False).get_opcodes() if kind!="equal"]
change_pass=(len(diff)==3 and
 diff[0]["kind"]=="insert" and diff[0]["new"][0].startswith("% v2.5.19") and
 diff[1]["kind"]=="replace" and all("pdftitle" in x for x in diff[1]["old"]+diff[1]["new"]) and
 diff[2]["kind"]=="replace" and "四十节" in " ".join(diff[2]["old"]) and
 "四十一节" in " ".join(diff[2]["new"]))
structure=inspect(B)
old_inputs=sorted(re.findall(r"\\(?:input|include)\s*\{([^}]+)\}",text(A)))
new_inputs=sorted(re.findall(r"\\(?:input|include)\s*\{([^}]+)\}",text(B)))
gate=(sha(A)==EXPECTED_A and sha(B)==EXPECTED_B and change_pass and
      old_inputs==new_inputs and
      structure["numbered_five_chapter_sections"]==41 and
      structure["unique_tex"]==36 and structure["input_edges"]==35 and
      not structure["missing_tex"] and not structure["duplicate_labels"] and
      not structure["unresolved_bibliography_keys"])
print(json.dumps({
 "source_old_sha":sha(A),"source_new_sha":sha(B),
 "diff_only_version_metadata_and_preface":change_pass,
 "root_tex_inputs_identical":old_inputs==new_inputs,
 "structural_audit":structure,"PASS":gate
},ensure_ascii=False,indent=2))
sys.exit(0 if gate else 1)
