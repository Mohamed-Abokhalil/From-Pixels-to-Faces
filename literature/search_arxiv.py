"""arXiv arm of the systematic search: title+abstract phrases of concept block A, cs.CV/cs.LG/eess.IV,
submitted 2016-01-01 .. search date. Writes literature/exports/arxiv_<date>.json."""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHRASES = ["face inpainting", "facial inpainting", "face completion", "facial image completion", "face image inpainting",
           "face de-occlusion", "face unmasking", "masked face reconstruction", "masked face restoration",
           "face mask removal", "occluded face completion", "facial occlusion removal",
           "facial image inpainting", "unmasking of masked face", "removal of face masks", "eye inpainting", "eye in-painting",
           "head inpainting", "masked face inpainting", "facial de-occlusion"]
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def main(d="2026-11-12"):
    terms = " OR ".join(f'ti:"{p}" OR abs:"{p}"' for p in PHRASES)
    q = f"({terms}) AND submittedDate:[201601010000 TO {d.replace('-', '')}2359]"
    recs, start = [], 0
    while True:
        url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": start, "max_results": 200, "sortBy": "submittedDate", "sortOrder": "ascending"})
        req = urllib.request.Request(url, headers={"User-Agent": "fpfs-revision-script/1.0"})
        xml = urllib.request.urlopen(req, timeout=120).read()
        root = ET.fromstring(xml)
        total = int(root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults").text)
        ents = root.findall("a:entry", NS)
        for e in ents:
            recs.append({"id": e.find("a:id", NS).text, "title": re.sub(r"\s+", " ", e.find("a:title", NS).text).strip(),
                         "abstract": re.sub(r"\s+", " ", e.find("a:summary", NS).text).strip(),
                         "published": e.find("a:published", NS).text[:10],
                         "doi": (e.find("arxiv:doi", NS).text if e.find("arxiv:doi", NS) is not None else None),
                         "journal_ref": (e.find("arxiv:journal_ref", NS).text if e.find("arxiv:journal_ref", NS) is not None else None),
                         "categories": [c.get("term") for c in e.findall("a:category", NS)]})
        print(len(recs), "/", total, flush=True)
        start += 200
        if start >= total or not ents:
            break
        time.sleep(3)
    json.dump({"query": q, "search_date": d, "n": len(recs), "records": recs},
              open(HERE / "exports" / f"arxiv_{d.replace('-', '')}.json", "w", encoding="utf8"))


if __name__ == "__main__":
    main(*sys.argv[1:])
