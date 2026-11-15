"""OpenAlex arm of the systematic search (title + abstract), 2016-01-01 .. search date.

Writes literature/exports/openalex_<date>.json (raw records) and prints the count.
No API key is used; requests carry a generic User-Agent only.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUERY = ('"face inpainting" OR "facial inpainting" OR "face completion" OR "facial image completion" OR '
         '"face image completion" OR "face image inpainting" OR "face de-occlusion" OR "face deocclusion" OR '
         '"face unmasking" OR "facial unmasking" OR "masked face reconstruction" OR "masked face restoration" OR '
         '"masked face completion" OR "masked face recovery" OR "face mask removal" OR "mask removal from face" OR '
         '"occluded face completion" OR "occluded face restoration" OR "occluded face reconstruction" OR '
         '"facial occlusion removal" OR "facial image inpainting" OR "facial image completion" OR "face image restoration from mask" OR '
         '"unmasking of masked face" OR "masked face unmasking" OR "removal of face masks" OR "remove face masks" OR '
         '"eye inpainting" OR "eye in-painting" OR "head inpainting" OR "face mask inpainting" OR "masked face inpainting" OR '
         '"facial de-occlusion" OR "face occlusion removal"')
FILTER = "title_and_abstract.search:{},from_publication_date:2016-01-01,to_publication_date:{}"
FIELDS = "id,doi,title,display_name,publication_year,publication_date,type,primary_location,authorships,abstract_inverted_index,cited_by_count"


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fpfs-script/1.0"})
    for i in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001
            print("retry", i, e, file=sys.stderr); time.sleep(5 * (i + 1))
    raise RuntimeError(url)


def main(search_date=None):
    d = search_date or date.today().isoformat()
    flt = FILTER.format(QUERY, d)
    page, recs = 1, []
    while True:  # basic paging (cursor paging requires an API key)
        url = ("https://api.openalex.org/works?" +
               urllib.parse.urlencode({"filter": flt, "per-page": 200, "page": page, "select": FIELDS}))
        j = get(url)
        recs += j["results"]
        print(len(recs), "/", j["meta"]["count"], flush=True)
        if not j["results"] or len(recs) >= j["meta"]["count"]:
            break
        page += 1; time.sleep(1)
    out = HERE / "exports" / f"openalex_{d.replace('-', '')}.json"
    json.dump({"query": QUERY, "filter": flt, "search_date": d, "n": len(recs), "records": recs},
              open(out, "w", encoding="utf8"))
    print("saved", out, len(recs))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
