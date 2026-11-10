#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only PDF preflight. No builds, writes, subprocesses or network.
Run: python3 script.py [PDF] [LOG]
Default checks PDF structure/links on all pages and textual sample only.
Never call "PASS" a full visual review; it is a structural gate.
"""
from pathlib import Path
from pypdf import PdfReader
import hashlib, json, sys, re
ROOT=Path(__file__).resolve().parent
DEFAULT=None
BASE_SHA=None
def secure_path(name):
    """Public-repo gate: any readable PDF/LOG path; no Mac-only jail."""
    p=Path(name)
    if not p.is_absolute():
        p=(Path.cwd()/p).resolve()
    else:
        p=p.resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return p
def digest(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
    return h.hexdigest()
def leaves(nodes):
    return sum(leaves(n) if isinstance(n,list) else 1 for n in nodes)
def check_pdf(p):
    reader=PdfReader(str(p))
    pages=len(reader.pages)
    links={"GoTo":0,"URI":0,"Other":0}
    warning=[]; geometry=[]
    for i,page in enumerate(reader.pages):
        try:
            box=page.mediabox
            w=float(box.width);h=float(box.height)
            if not (589 < w < 602 and 836 < h < 849):geometry.append(i+1)
            for ref in page.get("/Annots",[]):
                o=ref.get_object()
                act=o.get("/A")
                if act:
                    act=act.get_object(); typ=str(act.get("/S",""))
                    if typ=="/GoTo":
                        links["GoTo"]+=1
                        if act.get("/D") is None:warning.append(f"p{i+1}:GoTo_missing_target")
                    elif typ=="/URI":
                        links["URI"]+=1
                        if not act.get("/URI"):warning.append(f"p{i+1}:URI_missing")
                    else:links["Other"]+=1
                elif o.get("/Dest") is not None:links["GoTo"]+=1
                else:links["Other"]+=1
        except Exception as exc:warning.append(f"p{i+1}:{type(exc).__name__}:{exc}")
    # Deterministic sample across the volume: front, spaced interior, and end.
    sample=sorted(set([0,1,2,3,pages-4,pages-3,pages-2,pages-1]+
                      list(range(10,max(10,pages-10),max(10,pages//9)))))
    sample=[i for i in sample if 0<=i<pages]
    missing_text=[]
    for i in sample:
        try:
            if not (reader.pages[i].extract_text() or "").strip():missing_text.append(i+1)
        except Exception as exc:warning.append(f"p{i+1}:text:{type(exc).__name__}")
    with p.open("rb") as f:
        header=f.read(8);f.seek(max(0,p.stat().st_size-4096));tail=f.read()
    header_ok=header.startswith(b"%PDF-");eof_ok=b"%%EOF" in tail
    md=reader.metadata
    result={"path":str(p),"bytes":p.stat().st_size,"sha256":digest(p),"pages":pages,
       "bookmarks":leaves(reader.outline),"links":links,
       "metadata":{"title":str(md.title or ""),"author":str(md.author or "")},
       "header_ok":header_ok,"eof_ok":eof_ok,"sample_pages":[i+1 for i in sample],
       "sample_blank_text":missing_text,"non_a4_pages":geometry,"warnings":warning[:30]}
    result["structural_gate"]=bool(header_ok and eof_ok and pages>140 and
       result["bookmarks"]>=80 and links["GoTo"]>=180 and links["URI"]>=240 and
       not geometry and not warning and not missing_text)
    return result
def check_log(p):
    txt=p.read_text(encoding="utf-8",errors="replace")
    issues={
      "tex_fatal":len(re.findall(r"^! ",txt,re.M)),
      "undefined_references":len(re.findall(r"Reference .+ undefined|There were undefined references",txt)),
      "undefined_citations":len(re.findall(r"Citation .+ undefined",txt)),
      "overfull_hbox":len(re.findall(r"Overfull \\hbox",txt)),
      "underfull_hbox":len(re.findall(r"Underfull \\hbox",txt)),
      "rerun":len(re.findall(r"Rerun to get cross-references right",txt))}
    return {"path":str(p),"sha256":digest(p),"issues":issues,
         "reference_gate":not any(issues[k] for k in ("tex_fatal","undefined_references","undefined_citations"))}
def main():
    p=secure_path(sys.argv[1]) if len(sys.argv)>1 else DEFAULT
    result={"pdf":check_pdf(p)}
    if p==DEFAULT:result["expected_baseline_sha_match"]=result["pdf"]["sha256"]==BASE_SHA
    if len(sys.argv)>2:result["log"]=check_log(secure_path(sys.argv[2]))
    result["preflight_pass"]=result["pdf"]["structural_gate"] and result.get("log",{}).get("reference_gate",True)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["preflight_pass"] else 1
if __name__=="__main__":raise SystemExit(main())
