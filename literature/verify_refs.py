"""Verify bibliographic metadata via Crossref (by DOI, or by bibliographic query) and arXiv.

Input : a JSON list of {"label":..., "doi":... | "query":... | "arxiv":...}
Output: verified/<name>.json with the Crossref/arXiv record fields we use for BibTeX.
Only metadata returned by these services is written; nothing is filled in by hand.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "fpfs-revision-script/1.0"}


def get(url, raw=False):
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                b = r.read()
                return b if raw else json.loads(b)
        except Exception as e:  # noqa: BLE001
            print("  retry", i, e, file=sys.stderr); time.sleep(4 * (i + 1))
    return None


def crossref_item(it):
    return {"doi": it.get("DOI"), "title": " ".join(it.get("title", [""])), "container": " ".join(it.get("container-title", [""])),
            "type": it.get("type"), "year": (it.get("issued", {}).get("date-parts", [[None]])[0][0]),
            "volume": it.get("volume"), "issue": it.get("issue"), "pages": it.get("page") or it.get("article-number"),
            "publisher": it.get("publisher"),
            "authors": [f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in it.get("author", [])],
            "abstract": re.sub(r"<[^>]+>", "", it.get("abstract", ""))[:1500]}


def arxiv(aid):
    x = get("http://export.arxiv.org/api/query?id_list=" + aid, raw=True)
    if not x:
        return None
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    e = ET.fromstring(x).find("a:entry", ns)
    if e is None:
        return None
    g = lambda p: (e.find(p, ns).text if e.find(p, ns) is not None else None)
    return {"arxiv": aid, "title": re.sub(r"\s+", " ", g("a:title") or ""), "published": (g("a:published") or "")[:10],
            "journal_ref": g("x:journal_ref"), "doi": g("x:doi"),
            "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
            "abstract": re.sub(r"\s+", " ", g("a:summary") or "")[:1500]}


def main(inp, name):
    out = []
    for q in json.load(open(inp, encoding="utf8")):
        rec = {"label": q["label"]}
        if q.get("doi"):
            j = get("https://api.crossref.org/works/" + urllib.parse.quote(q["doi"]))
            rec["crossref"] = crossref_item(j["message"]) if j else None
        if q.get("query"):
            j = get("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(q["query"]))
            rec["crossref_candidates"] = [crossref_item(it) for it in (j or {}).get("message", {}).get("items", [])]
        if q.get("arxiv"):
            rec["arxiv"] = arxiv(q["arxiv"])
        out.append(rec)
        best = rec.get("crossref") or (rec.get("crossref_candidates") or [None])[0] or rec.get("arxiv")
        print(q["label"], "->", (best or {}).get("doi"), "|", (best or {}).get("title", "")[:70], "|",
              (best or {}).get("container", (best or {}).get("journal_ref")), (best or {}).get("year", (best or {}).get("published")))
        time.sleep(1)
    (HERE / "verified").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "verified" / f"{name}.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1], sys.argv[2])
