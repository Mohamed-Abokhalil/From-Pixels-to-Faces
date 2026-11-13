"""Copy generated table rows into sn-article.tex between marker lines (re-runnable).

First run: every \\inputrows{X} is replaced by
    %% BEGIN GENERATED X (benchmark_v2/make_tables.py or literature/make_taxonomy.py; do not edit by hand)
    ...content of X.tex...
    %% END GENERATED X
Later runs: the text between the markers is replaced by the current content of X.tex.
Keeps the submission a single .tex file and avoids \\input inside tables (breaks with adjustbox).
"""
import re
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "FI_FU_Survey_AIR_revised"
P = TEX / "sn-article.tex"
s = open(P, encoding="utf8").read()


def block(name):
    body = open(TEX / f"{name}.tex", encoding="utf8").read().rstrip("\n")
    return (f"%% BEGIN GENERATED {name} (benchmark_v2/make_tables.py or literature/make_taxonomy.py; do not edit by hand)\n"
            f"{body}\n%% END GENERATED {name}")


s, n1 = re.subn(r"\\inputrows\{([A-Za-z_]+)\}", lambda m: block(m.group(1)), s)
s, n2 = re.subn(r"%% BEGIN GENERATED ([A-Za-z_]+) .*?\n.*?%% END GENERATED \1", lambda m: block(m.group(1)), s, flags=re.S)
open(P, "w", encoding="utf8").write(s)
print("inlined", n1, "refreshed", n2)
