"""Generate the complete taxonomy table, the method counts and the venue distribution from
taxonomy_methods.csv. Checks every row against the bibliography.

Outputs (into manuscript/): tab_taxonomy_rows.tex, taxonomy_numbers.tex (\\newcommand macros)
and literature/taxonomy_checks.txt (venue/year cross-check against the .bib, missing keys).
"""
import csv
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "manuscript"
BIB = [TEX / "sn-bibliography.bib", HERE / "new_refs.bib"]

CAT = {"FI-specific": "FI", "FU-specific": "FU", "BFR (related)": "BFR", "General inpainting": "Gen"}
FAM = {"GAN": "GAN", "StyleGAN prior": "SG", "Codebook/Transformer": "TR/VQ", "Diffusion": "DM", "Transformer": "TR/VQ",
       "Unverified": "?"}
GUID = {"None": "--", "Landmark": "Lm", "Parsing": "Ps", "Segmentation": "Seg", "Attribute": "Att", "Reference": "Ref",
        "Sketch": "Sk", "Text": "Txt", "Audio": "Aud", "3D": "3D", "Edge": "Edge", "Expression": "Exp"}
SCEN = {"Irr": "Irr", "Comp": "Comp", "Lower": "Low", "Large": "Lg", "Real": "Real", "No mask": "--", "Object": "Obj",
        "Pose (UV)": "UV", "Comp (eyes)": "Comp", "Comp (mouth)": "Comp", "Comp (LR)": "Comp", "Irr;Pose": "Irr", "Real;Lower": "Real",
        "Irr;No mask": "Irr"}
EVAL = {"Dist": "D", "ID": "ID", "Div": "Dv", "User": "U"}


def bib():
    d = {}
    for p in BIB:
        if p.exists():
            for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", open(p, encoding="utf8").read(), re.S):
                body = m.group(3)
                g = lambda f: (re.search(rf"\b{f}\s*=\s*\{{(.*?)\}},?\s*\n", body, re.S) or [None, ""])[1]
                d[m.group(2).strip()] = {"type": m.group(1).lower(), "year": g("year"), "venue": g("journal") or g("booktitle") or g("howpublished")}
    return d


def main():
    rows = list(csv.DictReader(open(HERE / "taxonomy_methods.csv", encoding="utf8")))
    B = bib(); chk = []
    for r in rows:
        b = B.get(r["key"])
        if not b:
            chk.append(f"MISSING in bib: {r['key']}"); continue
        if b["year"] and b["year"][:4] != r["year"]:
            chk.append(f"YEAR {r['key']}: csv {r['year']} vs bib {b['year']}")
        if "arxiv" in b["venue"].lower() and r["venue_type"] != "P":
            chk.append(f"VENUE {r['key']}: bib says arXiv, csv venue_type {r['venue_type']}")
        chk.append(f"ok {r['key']}: csv '{r['venue']}' | bib '{b['venue'][:70]}'")
    open(HERE / "taxonomy_checks.txt", "w", encoding="utf8").write("\n".join(chk))
    rows.sort(key=lambda r: (int(r["year"]), r["method"].lower()))
    L = []
    for r in rows:
        g = ",".join(GUID.get(x, x) for x in r["guidance"].split(";"))
        ev = ",".join(EVAL.get(x, x) for x in r["evaluation"].split(";"))
        bench = ",".join(t.strip() for t in r["bench"].split(";") if t.strip() and "(excluded)" not in t) or "--"
        comp = r["component"].replace(";", "; ").replace("&", "\\&")
        L.append(f"{r['year']} & {r['method'].replace('&', chr(92) + '&')} \\cite{{{r['key']}}} & {r['venue_type']} & {CAT[r['category']]} & "
                 f"{FAM.get(r['family'], r['family'])} & {SCEN.get(r['scenario'], r['scenario'])} & {g} & "
                 f"{comp} & {ev} & {bench or '--'} \\\\")
    open(TEX / "tab_taxonomy_rows.tex", "w", encoding="utf8").write("\n".join(L) + "\n")
    c = Counter(r["category"] for r in rows); v = Counter(r["venue_type"] for r in rows)
    fam = Counter(r["family"] for r in rows if r["category"] in ("FI-specific", "FU-specific"))
    face = [r for r in rows if r["category"] in ("FI-specific", "FU-specific")]
    vf = Counter(r["venue_type"] for r in face)
    words = {"FI-specific": "FIspecific", "FU-specific": "FUspecific", "BFR (related)": "BFR", "General inpainting": "General"}
    N = {"TaxTotal": len(rows), "TaxFace": len(face), "TaxJournal": v["J"], "TaxConference": v["C"], "TaxPreprint": v["P"],
         "TaxFaceJournal": vf["J"], "TaxFaceConference": vf["C"], "TaxFacePreprint": vf["P"],
         "TaxFamGAN": fam["GAN"], "TaxFamSG": fam["StyleGAN prior"], "TaxFamTR": fam["Codebook/Transformer"] + fam["Transformer"],
         "TaxFamDM": fam["Diffusion"], "TaxFamUnverified": fam["Unverified"],
         "TaxNew": sum(1 for r in rows if r["route"] in ("search", "suggested")) }
    for k, w in words.items():
        N["Tax" + w] = c[k]
    with open(TEX / "taxonomy_numbers.tex", "w") as fh:
        for k, val in N.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{val}}}\n")
    print(N); print("checks:", sum(1 for x in chk if not x.startswith("ok")), "issues")
    for x in chk:
        if not x.startswith("ok"):
            print(" ", x)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
