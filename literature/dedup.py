"""Merge all exports in literature/exports/ and remove duplicates.

Duplicate rule: same DOI (case-insensitive), else same normalised title (lower-case, alphanumerics only).
When an arXiv preprint and a peer-reviewed version coexist, the peer-reviewed record is kept and the
preprint is logged as a duplicate of it. Supports OpenAlex JSON, arXiv JSON, and (when provided)
Scopus CSV / IEEE CSV / WoS tab-delimited exports.
Output: screening/records_dedup.csv, screening/duplicates.csv, screening/counts_identification.json
"""
import csv
import glob
import json
import re
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "screening"; OUT.mkdir(exist_ok=True)


def norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def abstract_from_inv(inv):
    if not inv:
        return ""
    pos = {}
    for w, idx in inv.items():
        for i in idx:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


def load():
    recs, per_source = [], {}
    for p in sorted(glob.glob(str(HERE / "exports" / "*"))):
        name = Path(p).name
        if name.startswith("openalex_"):
            for r in json.load(open(p, encoding="utf8"))["records"]:
                loc = r.get("primary_location") or {}
                src = (loc.get("source") or {})
                recs.append({"source_db": "OpenAlex", "src_id": r["id"], "doi": (r.get("doi") or "").replace("https://doi.org/", "").lower(),
                             "title": r.get("display_name") or r.get("title") or "", "year": r.get("publication_year"),
                             "type": r.get("type"), "venue": src.get("display_name") or "", "venue_type": src.get("type") or "",
                             "first_author": (r.get("authorships") or [{}])[0].get("author", {}).get("display_name", "") if r.get("authorships") else "",
                             "abstract": abstract_from_inv(r.get("abstract_inverted_index"))})
        elif name.startswith("arxiv_"):
            for r in json.load(open(p, encoding="utf8"))["records"]:
                recs.append({"source_db": "arXiv", "src_id": r["id"], "doi": (r.get("doi") or "").lower(), "title": r["title"],
                             "year": int(r["published"][:4]), "type": "preprint", "venue": r.get("journal_ref") or "arXiv",
                             "venue_type": "repository", "first_author": "", "abstract": r["abstract"]})
        elif name.startswith("scopus_") and name.endswith(".csv"):
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                recs.append({"source_db": "Scopus", "src_id": r.get("EID", ""), "doi": (r.get("DOI") or "").lower(), "title": r.get("Title", ""),
                             "year": int(r.get("Year") or 0), "type": r.get("Document Type", ""), "venue": r.get("Source title", ""),
                             "venue_type": r.get("Document Type", ""), "first_author": (r.get("Authors") or "").split(";")[0],
                             "abstract": r.get("Abstract", "")})
        elif name.startswith("ieee_") and name.endswith(".csv"):
            for r in csv.DictReader(open(p, encoding="utf-8-sig")):
                recs.append({"source_db": "IEEE Xplore", "src_id": r.get("PDF Link", ""), "doi": (r.get("DOI") or "").lower(),
                             "title": r.get("Document Title", ""), "year": int(r.get("Publication Year") or 0), "type": r.get("Document Identifier", ""),
                             "venue": r.get("Publication Title", ""), "venue_type": r.get("Document Identifier", ""),
                             "first_author": (r.get("Authors") or "").split(";")[0], "abstract": r.get("Abstract", "")})
        elif name.startswith("wos_"):
            for r in csv.DictReader(open(p, encoding="utf-8-sig"), delimiter="\t"):
                recs.append({"source_db": "WoS", "src_id": r.get("UT", ""), "doi": (r.get("DI") or "").lower(), "title": r.get("TI", ""),
                             "year": int(r.get("PY") or 0), "type": r.get("DT", ""), "venue": r.get("SO", ""), "venue_type": r.get("DT", ""),
                             "first_author": (r.get("AU") or "").split(";")[0], "abstract": r.get("AB", "")})
        else:
            continue
        per_source[name] = sum(1 for x in recs if x["src_id"]) - sum(per_source.values())
    return recs, per_source


def main():
    recs, per_source = load()
    keep, dups = OrderedDict(), []
    rank = {"Scopus": 0, "WoS": 1, "IEEE Xplore": 2, "OpenAlex": 3, "arXiv": 4}
    recs.sort(key=lambda r: (r["type"] == "preprint", rank.get(r["source_db"], 9)))
    by_doi, by_title = {}, {}
    for r in recs:
        k = None
        if r["doi"] and r["doi"] in by_doi:
            k = by_doi[r["doi"]]
        elif norm(r["title"]) in by_title:
            k = by_title[norm(r["title"])]
        if k is not None:
            dups.append({"dup_src": r["source_db"], "dup_id": r["src_id"], "dup_title": r["title"], "kept_id": k})
            keep[k]["found_in"] += ";" + r["source_db"]
            continue
        rid = f"R{len(keep) + 1:04d}"
        r = dict(r, rec_id=rid, found_in=r["source_db"])
        keep[rid] = r
        if r["doi"]:
            by_doi[r["doi"]] = rid
        by_title[norm(r["title"])] = rid
    fields = ["rec_id", "found_in", "source_db", "src_id", "doi", "year", "type", "venue", "venue_type", "first_author", "title", "abstract"]
    with open(OUT / "records_dedup.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader(); w.writerows(keep.values())
    with open(OUT / "duplicates.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dup_src", "dup_id", "dup_title", "kept_id"]); w.writeheader(); w.writerows(dups)
    counts = {"per_export": per_source, "identified_total": len(recs), "duplicates_removed": len(dups), "after_dedup": len(keep)}
    json.dump(counts, open(OUT / "counts_identification.json", "w"), indent=1)
    print(counts)


if __name__ == "__main__":
    main()
