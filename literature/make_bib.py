"""Build BibTeX entries for new references from Crossref (DOI) / arXiv metadata only.

Input: new_refs_list.json  [{"key":..., "doi":...} | {"key":..., "arxiv":...}]
Output: new_refs.bib (appended to the manuscript bibliography) and new_refs_meta.json (raw metadata, audit trail)
"""
import json
import re
import sys
import urllib.parse

from verify_refs import arxiv, crossref_item, get

HERE = __import__("pathlib").Path(__file__).resolve().parent


def esc(s):
    s = re.sub(r"\s+", " ", s or "").strip()
    s = s.replace("&amp;", "&").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
    return s


def bib_from_crossref(key, c):
    typ = {"journal-article": "article", "proceedings-article": "inproceedings", "book-chapter": "incollection",
           "posted-content": "misc"}.get(c["type"], "misc")
    f = {"title": "{" + esc(c["title"]) + "}", "author": " and ".join(esc(a) for a in c["authors"]), "year": str(c["year"]),
         "doi": c["doi"]}
    if typ == "article":
        f["journal"] = esc(c["container"])
    elif typ in ("inproceedings", "incollection"):
        f["booktitle"] = esc(c["container"])
    else:
        f["howpublished"] = esc(c["container"] or c.get("publisher") or "")
    for k in ("volume", "issue", "pages"):
        if c.get(k):
            f["number" if k == "issue" else k] = esc(str(c[k])).replace("-", "--") if k == "pages" else esc(str(c[k]))
    if c.get("publisher"):
        f["publisher"] = esc(c["publisher"])
    return f"@{typ}{{{key},\n" + ",\n".join(f"  {k}={{{v}}}" if not v.startswith("{") else f"  {k}={v}" for k, v in f.items()) + "\n}\n"


def bib_from_arxiv(key, a):
    f = {"title": "{" + esc(a["title"]) + "}", "author": " and ".join(esc(x) for x in a["authors"]),
         "year": a["published"][:4], "eprint": a["arxiv"], "archivePrefix": "arXiv", "howpublished": f"arXiv:{a['arxiv']}",
         "note": "Preprint"}
    return f"@misc{{{key},\n" + ",\n".join(f"  {k}={{{v}}}" if not v.startswith("{") else f"  {k}={v}" for k, v in f.items()) + "\n}\n"


def main():
    items = json.load(open(HERE / "new_refs_list.json", encoding="utf8"))
    out, meta = [], []
    for it in items:
        if it.get("doi"):
            j = get("https://api.crossref.org/works/" + urllib.parse.quote(it["doi"]))
            if not j:
                print("FAILED", it); continue
            c = crossref_item(j["message"]); meta.append({"key": it["key"], "crossref": c})
            out.append(bib_from_crossref(it["key"], c))
        else:
            a = arxiv(it["arxiv"])
            if not a:
                print("FAILED", it); continue
            meta.append({"key": it["key"], "arxiv": a}); out.append(bib_from_arxiv(it["key"], a))
        print("ok", it["key"])
    open(HERE / "new_refs.bib", "w", encoding="utf8").write("\n".join(out))
    json.dump(meta, open(HERE / "new_refs_meta.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
