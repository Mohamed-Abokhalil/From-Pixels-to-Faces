"""Generate every benchmark table/number used in the manuscript from stored per-image files.

Inputs : results/<task>/per_image/*.csv, results/<task>/fid_kid.csv, results/<task>/arcface_per_image.csv,
         results/stoch/*.csv (when present), the submitted summary CSVs (for the old-vs-new comparison)
Outputs: ../manuscript/tab_fi_bench.tex, tab_fu_bench.tex, tab_fi_bins.tex   (\\input in the paper)
         results/<task>/summary.csv, results/old_vs_new_<task>.csv, results/stats_<task>.json, results/numbers.tex
         (numbers.tex defines \\newcommand macros for every number quoted in the text)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from common import METHODS, RESULTS, ROOT

HERE = Path(__file__).resolve().parent
TEX = HERE.parent / "manuscript"
OLD = {"fi": ROOT / "results" / "fi_celeba" / "summary_by_bin.csv", "fu": ROOT / "results_bin" / "fu_celeba" / "summary_by_bin.csv"}
OLD_ID = {"fi": ROOT / "results" / "fi_arcface.json", "fu": ROOT / "results" / "fu_arcface.json"}
FOLDER_ALIAS = {"DeepFillv2": ["deepfillv2", "DeepFillv2"], "EdgeConnect": ["EdgConnect", "EdgeConnect"]}
NUM = {}  # macro name -> value string


def macro(name, val):
    NUM[name] = val
    return val


def per_image(task, folder):
    p = RESULTS / task / "per_image" / f"{folder}.csv"
    return list(csv.DictReader(open(p))) if p.exists() else None


def arc(task):
    p = RESULTS / task / "arcface_per_image.csv"
    d = defaultdict(dict)
    if p.exists():
        for r in csv.DictReader(open(p)):
            d[(r["folder"], r["kind"])][r["file"]] = (float(r["idsim_legacy"]),
                                                      float(r["idsim_aligned"]) if r["idsim_aligned"] else np.nan)
    return d


def fidk(task):
    p = RESULTS / task / "fid_kid.csv"
    return {(r["folder"], r["kind"]): r for r in csv.DictReader(open(p))} if p.exists() else {}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(task):
    A, F = arc(task), fidk(task)
    rows, per = [], {}
    refs = [("Visible-only (mean fill)", "_meanfill"), ("Masked input", "_masked")]
    for m in METHODS[task]:
        pi = per_image(task, m.folder)
        if pi is None:
            continue
        r = {"method": m.name, "folder": m.folder, "year": m.year, "cite": m.cite, "category": m.category, "family": m.family,
             "team": m.team, "status": m.status, "n": len(pi)}
        for k in ("psnr_full", "ssim_full", "l1_full", "psnr_hole", "ssim_hole", "l1_hole", "psnr_band", "ssim_band", "l1_band", "lpips_full"):
            r["comp_" + k] = float(np.mean([float(x["comp_" + k]) for x in pi]))
            r["raw_" + k] = float(np.mean([float(x["raw_" + k]) for x in pi]))
        r["vis_err_raw_255"] = 255 * float(np.mean([float(x["vis_abs_err_raw"]) for x in pi]))
        r["psnr_hole_ci"] = boot_ci([float(x["comp_psnr_hole"]) for x in pi])
        for kind in ("comp", "raw"):
            f = F.get((m.folder, kind))
            if f:
                r[f"{kind}_fid"] = float(f["fid"]); r[f"{kind}_kid"] = float(f["kid_x1e3"]); r[f"{kind}_kid_std"] = float(f["kid_std_x1e3"])
                if f.get("fid_boot_sd"):
                    r[f"{kind}_fid_sd"] = float(f["fid_boot_sd"])
            a = A.get((m.folder, kind))
            if a:
                r[f"{kind}_id_legacy"] = float(np.mean([v[0] for v in a.values()]))
                r[f"{kind}_id_aligned"] = float(np.nanmean([v[1] for v in a.values()]))
                if kind == "comp":
                    r["id_aligned_ci"] = boot_ci([v[1] for v in a.values()])
        per[m.folder] = pi
        rows.append(r)
    ref_rows = []
    for name, folder in refs:
        pi = per_image(task, folder)
        if pi is None:
            continue
        r = {"method": name, "folder": folder, "status": "reference", "n": len(pi)}
        for k in ("psnr_full", "ssim_full", "l1_full", "psnr_hole", "ssim_hole", "l1_hole", "lpips_full"):
            r["comp_" + k] = float(np.mean([float(x["raw_" + k]) for x in pi]))
        a = A.get((folder, "raw"))
        if a:
            r["comp_id_aligned"] = float(np.nanmean([v[1] for v in a.values()]))
            r["comp_id_legacy"] = float(np.mean([v[0] for v in a.values()]))
        ref_rows.append(r)
    return rows, ref_rows, per, A


def fmt(v, d):
    return "--" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def best_marks(rows, key, higher):
    vals = [(r.get(key), i) for i, r in enumerate(rows) if r["status"] != "excluded" and r.get(key) is not None]
    vals.sort(key=lambda t: -t[0] if higher else t[0])
    return {vals[0][1]: "b", vals[1][1]: "u"} if len(vals) > 1 else {}


def cell(r, key, d, marks, i):
    s = fmt(r.get(key), d)
    return {"b": f"\\textbf{{{s}}}", "u": f"\\underline{{{s}}}"}.get(marks.get(i), s)


def bench_table(task, rows, refs):
    rows = sorted(rows, key=lambda r: (r["year"], r["method"]))
    cols = [("comp_psnr_full", 2, True), ("comp_psnr_hole", 2, True), ("comp_ssim_hole", 3, True),
            ("comp_lpips_full", 4, False), ("comp_fid", 2, False), ("comp_kid", 2, False), ("comp_id_aligned", 3, True)]
    marks = {k: best_marks(rows, k, h) for k, _, h in cols}
    cat = {"FI-specific": "FI", "FU-specific": "FU", "BFR (related)": "BFR", "General inpainting": "Gen"}
    L = []
    for i, r in enumerate(rows):
        name = r["method"] + ("$^\\dagger$" if r["team"] else "") + ("$^\\ast$" if r["status"] == "rerun" else "")
        if r["status"] == "excluded":
            continue
        sd = r.get("comp_fid_sd")
        fid = cell(r, "comp_fid", 2, marks["comp_fid"], i)   # bootstrap s.d. range given in the caption (macros below)
        L.append(f"{r['year']} & {name}~\\cite{{{r['cite']}}} & {cat[r['category']]} & " +
                 " & ".join(fid if k == "comp_fid" else cell(r, k, d, marks[k], i) for k, d, _ in cols) + " \\\\")
    L.append("\\midrule")
    for r in refs:
        L.append(f"-- & \\textit{{{r['method']}}} & -- & " + " & ".join(fmt(r.get(k), d) if k not in ("comp_fid", "comp_kid") else "--" for k, d, _ in cols) + " \\\\")
    open(TEX / f"tab_{task}_bench_rows.tex", "w", encoding="utf8").write("\n".join(L) + "\n")
    sds = [r["comp_fid_sd"] for r in rows if r["status"] != "excluded" and r.get("comp_fid_sd") is not None]
    if sds:
        macro(f"{task.upper()}fidSDmin", f"{min(sds):.1f}"); macro(f"{task.upper()}fidSDmax", f"{max(sds):.1f}")


def bins_table(rows, per, A):
    import csv as _csv
    ts = {r["file"]: r["bin"] for r in _csv.DictReader(open(HERE / "testsets" / "fi_testset.csv"))}
    order = ["0-10", "10-20", "20-30", "30-40", "40-50", ">50"]
    n = {b: sum(1 for v in ts.values() if v == b) for b in order}
    keep = [r for r in sorted(rows, key=lambda r: -r["comp_psnr_hole"]) if r["status"] != "excluded"]
    L = []
    for r in keep:
        pi = per[r["folder"]]; a = A.get((r["folder"], "comp"), {})
        ph = [np.mean([float(x["comp_psnr_hole"]) for x in pi if ts[x["file"]] == b]) for b in order]
        idv = [np.nanmean([a[f][1] for f in a if ts[f] == b]) if a else np.nan for b in order]
        L.append(f"{r['method']} & " + " & ".join(f"{p:.1f}/{q:.3f}".replace("/0.", "/.") for p, q in zip(ph, idv)) + " \\\\")
    hdr = " & ".join(f"{b.replace('>', '$>$')}\\% ({n[b]})" for b in order)
    open(TEX / "tab_fi_bins_rows.tex", "w", encoding="utf8").write("\n".join(L) + "\n")
    open(TEX / "tab_fi_bins_header.tex", "w", encoding="utf8").write("Method & " + hdr + " \\\\\n")
    for b, letter in zip(order, "ABCDEF"):   # LaTeX macro names cannot contain digits: bin A = 0-10% ... F = >50%
        macro(f"FIbinN{letter}", str(n[b]))


def subset_check(rows, per, A):
    """Possible train/test overlap: FI files are numbered 25001-29993 (CelebA-HQ-like indices); many CelebA-HQ checkpoints
    train on indices < 28000. Re-rank methods on the images numbered >= 28000 and compare with the full-set ranking."""
    ok = [r for r in rows if r["status"] != "excluded"]
    keep = lambda f: int(f[:-4]) >= 28000
    out = []
    for r in ok:
        pi = [x for x in per[r["folder"]] if keep(x["file"])]
        a = A.get((r["folder"], "comp"), {})
        out.append({"method": r["method"], "n": len(pi), "psnr_hole_full": r["comp_psnr_hole"],
                    "psnr_hole_sub": float(np.mean([float(x["comp_psnr_hole"]) for x in pi])),
                    "id_full": r.get("comp_id_aligned"), "id_sub": float(np.nanmean([a[f][1] for f in a if keep(f)])) if a else np.nan})
    with open(RESULTS / "fi_subset_ge28000.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    rho_p = spearmanr([o["psnr_hole_full"] for o in out], [o["psnr_hole_sub"] for o in out]).correlation
    rho_i = spearmanr([o["id_full"] for o in out], [o["id_sub"] for o in out]).correlation
    best_full = max(out, key=lambda o: o["psnr_hole_full"])["method"]; best_sub = max(out, key=lambda o: o["psnr_hole_sub"])["method"]
    macro("FIsubN", str(out[0]["n"])); macro("FIsubRhoPSNR", f"{rho_p:.2f}"); macro("FIsubRhoID", f"{rho_i:.2f}")
    macro("FIsubBestMethod", best_sub)
    gap = {o["method"]: o["psnr_hole_sub"] - o["psnr_hole_full"] for o in out}
    macro("FIsubGapMin", f"{min(gap.values()):.2f}"); macro("FIsubGapMax", f"{max(gap.values()):.2f}")
    print("subset >=28000:", out[0]["n"], "rho psnr", round(rho_p, 3), "rho id", round(rho_i, 3), "best full", best_full, "best sub", best_sub,
          "gap range", round(min(gap.values()), 2), round(max(gap.values()), 2))


def old_vs_new(task, rows):
    old = {r["method"]: r for r in csv.DictReader(open(OLD[task]))}
    oid = {r["method"]: r for r in json.load(open(OLD_ID[task]))}
    out = []
    for r in rows:
        keys = FOLDER_ALIAS.get(r["folder"], [r["folder"]]) + [r["method"]]
        o = next((old[k] for k in keys if k in old), None); oi = next((oid[k] for k in keys if k in oid), None)
        out.append({"method": r["method"], "status": r["status"],
                    "old_psnr": o and float(o["psnr"]), "new_raw_psnr_full": r["raw_psnr_full"], "new_comp_psnr_full": r["comp_psnr_full"],
                    "new_comp_psnr_hole": r["comp_psnr_hole"], "old_ssim": o and float(o["ssim"]), "new_comp_ssim_full": r["comp_ssim_full"],
                    "old_fid": o and float(o["fid"]), "new_raw_fid": r.get("raw_fid"), "new_comp_fid": r.get("comp_fid"),
                    "old_lpips": o and float(o["lpips"]), "new_comp_lpips": r["comp_lpips_full"],
                    "old_id": oi and oi["id_sim_mean"], "new_raw_id_legacy": r.get("raw_id_legacy"), "new_comp_id_aligned": r.get("comp_id_aligned")})
    with open(RESULTS / f"old_vs_new_{task}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    return out


def stats(task, rows, per, A, refs):
    ok = [r for r in rows if r["status"] != "excluded"]
    top = sorted(ok, key=lambda r: -r["comp_psnr_hole"])[:4]
    res = {"top_by_hole_psnr": [r["method"] for r in top], "pairwise": []}
    pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
    pv = []
    for a, b in pairs:
        x = np.array([float(v["comp_psnr_hole"]) for v in per[a["folder"]]]); y = np.array([float(v["comp_psnr_hole"]) for v in per[b["folder"]]])
        p = wilcoxon(x, y).pvalue
        pv.append(p); res["pairwise"].append({"a": a["method"], "b": b["method"], "metric": "psnr_hole", "median_diff": float(np.median(x - y)), "p": float(p)})
    order = np.argsort(pv); m = len(pv); adj = [0] * m; run = 0
    for k, idx in enumerate(order):
        run = max(run, min(1.0, (m - k) * pv[idx])); adj[idx] = run
    for d, a_ in zip(res["pairwise"], adj):
        d["p_holm"] = float(a_)
    # metric agreement: Spearman between per-image metrics pooled over methods
    pool = defaultdict(list)
    for r in ok:
        a = A.get((r["folder"], "comp"), {})
        for v in per[r["folder"]]:
            if a and v["file"] in a and not np.isnan(a[v["file"]][1]):
                pool["psnr_hole"].append(float(v["comp_psnr_hole"])); pool["lpips"].append(float(v["comp_lpips_full"]))
                pool["ssim_full"].append(float(v["comp_ssim_full"])); pool["id"].append(a[v["file"]][1])
    res["spearman"] = {f"{x}~{y}": float(spearmanr(pool[x], pool[y]).correlation)
                       for x, y in (("psnr_hole", "id"), ("lpips", "id"), ("psnr_hole", "lpips"), ("ssim_full", "id"))}
    # method-level rank agreement
    def rk(key, hi=True):
        v = [r.get(key, np.nan) for r in ok]; return np.array(v) * (1 if hi else -1)
    res["method_rank_spearman"] = {
        "psnr_hole~id_aligned": float(spearmanr(rk("comp_psnr_hole"), rk("comp_id_aligned")).correlation),
        "fid~id_aligned": float(spearmanr(rk("comp_fid", False), rk("comp_id_aligned")).correlation),
        "legacy_id~aligned_id": float(spearmanr(rk("raw_id_legacy"), rk("comp_id_aligned")).correlation)}
    vis = next((r for r in refs if r["folder"] == "_meanfill"), None)
    if vis and vis.get("comp_id_aligned") is not None:
        res["visible_only_id_aligned"] = vis["comp_id_aligned"]
        res["id_gain_over_visible_only"] = {r["method"]: r.get("comp_id_aligned", np.nan) - vis["comp_id_aligned"] for r in ok}
    json.dump(res, open(RESULTS / f"stats_{task}.json", "w"), indent=1)
    return res


def main():
    for task in ("fi", "fu"):
        rows, refs, per, A = summarize(task)
        with open(RESULTS / task / "summary.csv", "w", newline="") as fh:
            flat = [{k: (json.dumps(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows + refs]
            keys = sorted({k for r in flat for k in r})
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(flat)
        bench_table(task, rows, refs)
        old_vs_new(task, rows)
        st = stats(task, rows, per, A, refs)
        T = task.upper()
        ok = [r for r in rows if r["status"] != "excluded"]
        macro(f"{T}nMethods", str(len(ok)))
        macro(f"{T}nImages", str(rows[0]["n"]))
        for key, nm, hi in (("comp_psnr_hole", "HolePSNR", True), ("comp_id_aligned", "IDAligned", True), ("comp_fid", "FID", False),
                            ("comp_lpips_full", "LPIPS", False)):
            c = [r for r in ok if r.get(key) is not None]
            b = max(c, key=lambda r: r[key]) if hi else min(c, key=lambda r: r[key])
            macro(f"{T}best{nm}Method", b["method"]); macro(f"{T}best{nm}", f"{b[key]:.3f}" if "ID" in nm or "LPIPS" in nm else f"{b[key]:.2f}")
        if "visible_only_id_aligned" in st:
            macro(f"{T}visOnlyID", f"{st['visible_only_id_aligned']:.3f}")
        for k, v in st["spearman"].items():
            macro(f"{T}rho" + k.replace("~", "Vs").replace("_", ""), f"{v:.2f}")
        for k, v in st["method_rank_spearman"].items():
            macro(f"{T}rankrho" + k.replace("~", "Vs").replace("_", ""), f"{v:.2f}")
        ts = [float(r["mask_ratio"]) for r in csv.DictReader(open(HERE / "testsets" / f"{task}_testset.csv"))]
        macro(f"{T}maskMean", f"{100 * np.mean(ts):.1f}"); macro(f"{T}maskMin", f"{100 * min(ts):.1f}")
        macro(f"{T}maskMax", f"{100 * max(ts):.1f}")
        if task == "fi":
            bins_table(rows, per, A)
            subset_check(rows, per, A)
    # derived macros used in the results text (ranks are by recomposited hole PSNR / aligned ID, excluded methods dropped)
    for task in ("fi", "fu"):
        T = task.upper()
        rows, refs, per, A = summarize(task)
        ok = [r for r in rows if r["status"] != "excluded"]
        byh = sorted(ok, key=lambda r: -r["comp_psnr_hole"]); byi = sorted(ok, key=lambda r: -r["comp_id_aligned"])
        rank_h = {r["method"]: i + 1 for i, r in enumerate(byh)}; rank_i = {r["method"]: i + 1 for i, r in enumerate(byi)}
        num = lambda i: ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
                         "twelfth"][i - 1] if i <= 12 else f"{i}th"
        macro(f"{T}secondHoleMethod", byh[1]["method"]); macro(f"{T}secondHole", f"{byh[1]['comp_psnr_hole']:.2f}")
        macro(f"{T}holeMin", f"{byh[-1]['comp_psnr_hole']:.2f}"); macro(f"{T}holeMax", f"{byh[0]['comp_psnr_hole']:.2f}")
        fulls = [r["comp_psnr_full"] for r in ok]; macro(f"{T}fullMin", f"{min(fulls):.2f}"); macro(f"{T}fullMax", f"{max(fulls):.2f}")
        ids = [r["comp_id_aligned"] for r in ok]; macro(f"{T}idMin", f"{min(ids):.3f}"); macro(f"{T}idMax", f"{max(ids):.3f}")
        vis = next(r for r in refs if r["folder"] == "_meanfill")["comp_id_aligned"]
        macro(f"{T}nBelowVisID", str(sum(r["comp_id_aligned"] < vis for r in ok)))
        top4 = byh[:4]; macro(f"{T}topFourHole", ", ".join(r["method"] for r in top4))
        macro(f"{T}topFourAllGeneral", "yes" if all(r["category"] == "General inpainting" for r in top4) else "no")
        fis = [r for r in byh if r["category"] == "FI-specific"]
        if fis:
            macro(f"{T}bestFIspecMethod", fis[0]["method"]); macro(f"{T}bestFIspecRank", num(rank_h[fis[0]["method"]]))
            macro(f"{T}bestFIspecHole", f"{fis[0]['comp_psnr_hole']:.2f}")
        dm = [r for r in byh if r["family"] == "Diffusion"]
        if dm:
            macro(f"{T}bestDiffMethod", dm[0]["method"]); macro(f"{T}bestDiffHoleRank", num(rank_h[dm[0]["method"]]))
            macro(f"{T}nAheadOfDiff", str(rank_h[dm[0]["method"]] - 1))
            bdi = min(dm, key=lambda r: rank_i[r["method"]]); macro(f"{T}bestDiffIDMethod", bdi["method"]); macro(f"{T}bestDiffIDRank", num(rank_i[bdi["method"]]))
        bfr = [r for r in byh if r["category"].startswith("BFR")]
        if bfr:
            macro(f"{T}bfrRanks", " and ".join(f"{r['method']} {num(rank_h[r['method']])}" for r in bfr))
        macro(f"{T}nOK", str(len(ok)))
        macro(f"{T}nExcluded", str(sum(1 for r in rows if r["status"] == "excluded")))
        if task == "fu":
            din = next((r for r in ok if r["folder"] == "Din2020"), None)   # excluded since the official-checkpoint rule
            if din:
                macro("FUdinRank", num(rank_h[din["method"]]))
                macro("FUnGenAboveDin", str(sum(1 for r in ok if r["category"] == "General inpainting" and r["comp_psnr_hole"] > din["comp_psnr_hole"])))
            macro("FUnSpecific", str(sum(1 for r in ok if r["category"] == "FU-specific")))
            macro("FUnGeneral", str(sum(1 for r in ok if r["category"] == "General inpainting")))
            for folder, nm in (("GANMasker", "gan"), ("M2UNet", "mtwo")):   # side-by-side with the values in the papers
                t = next(r for r in ok if r["folder"] == folder)
                macro(f"FU{nm}PSNRfull", f"{t['comp_psnr_full']:.2f}"); macro(f"FU{nm}SSIMfull", f"{t['comp_ssim_full']:.3f}")
                macro(f"FU{nm}FID", f"{t['comp_fid']:.2f}")
            st = json.load(open(RESULTS / "stats_fu.json")); pr = st["pairwise"][0]
            macro("FUtopPairDiff", f"{pr['median_diff']:.2f}"); macro("FUtopPairA", pr["a"]); macro("FUtopPairB", pr["b"])
            macro("FUvisOnlyGapBest", f"{vis - byi[0]['comp_id_aligned']:.3f}")
    ov = {r["method"]: r for r in csv.DictReader(open(RESULTS / "old_vs_new_fi.csv"))}
    macro("TSFold", f"{float(ov['TSFormer']['old_psnr']):.2f}"); macro("TSFnew", f"{float(ov['TSFormer']['new_comp_psnr_full']):.2f}")
    macro("PLold", f"{float(ov['Pluralistic']['old_psnr']):.2f}"); macro("PLnew", f"{float(ov['Pluralistic']['new_comp_psnr_full']):.2f}")
    macro("SPold", f"{float(ov['Spa-former']['old_psnr']):.2f}"); macro("SPnew", f"{float(ov['Spa-former']['new_comp_psnr_full']):.2f}")
    mp = HERE / "phase0_audit" / "memorization_probe.json"
    if mp.exists():
        d = json.load(open(mp))
        macro("MemN", str(d["Spa-former_fi"]["n"]))
        macro("MemSpaFI", f"{d['Spa-former_fi']['lower_half_psnr_mean']:.1f}"); macro("MemSpaFU", f"{d['Spa-former_fu']['lower_half_psnr_mean']:.1f}")
        macro("MemVhiiFI", f"{d['VHII_fi']['lower_half_psnr_mean']:.1f}"); macro("MemVhiiFU", f"{d['VHII_fu']['lower_half_psnr_mean']:.1f}")
    # repeated-sampling macros (results/stoch/<task>_<Method>.json from eval_stoch.py)
    words = {"fi": "FI", "fu": "FU", "Pluralistic": "Plur", "RePaint": "Rep"}
    for p in sorted((RESULTS / "stoch").glob("*.json")):
        d = json.load(open(p)); key = "St" + words[d["task"]] + words.get(d["method"], d["method"])
        macro(key + "N", str(d["n_images"])); macro(key + "K", str(d["K"]))
        macro(key + "PSNR", f"{d['psnr_hole']['mean']:.2f}"); macro(key + "PSNRrunSD", f"{d['psnr_hole']['run_mean_std']:.2f}")
        macro(key + "PSNRimgSD", f"{d['psnr_hole']['per_image_std_mean']:.2f}")
        macro(key + "ID", f"{d['id_aligned']['mean']:.3f}"); macro(key + "IDimgSD", f"{d['id_aligned']['per_image_std_mean']:.3f}")
        macro(key + "IDrunSD", f"{d['id_aligned']['run_mean_std']:.3f}"); macro(key + "Div", f"{d['diversity_pairwise_lpips']:.4f}")
    # pairwise tests of the leading methods
    for task in ("fi", "fu"):
        st = json.load(open(RESULTS / f"stats_{task}.json"))
        macro(f"{task.upper()}maxPholm", f"{max(p['p_holm'] for p in st['pairwise']):.1e}".replace("e-", "\\times10^{-").replace("e+", "\\times10^{") + "}")
    with open(RESULTS / "numbers.tex", "w") as fh:
        for k, v in sorted(NUM.items()):
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    import shutil
    shutil.copy(RESULTS / "numbers.tex", TEX / "bench_numbers.tex")
    print(json.dumps(NUM, indent=0))


if __name__ == "__main__":
    main()
