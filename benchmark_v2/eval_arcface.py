"""ArcFace identity similarity (ID-Sim) under two protocols.

Recognition model (both protocols): InsightFace pack buffalo_l, file w600k_r50.onnx
(ResNet-50 trained on WebFace600K, per the official InsightFace model zoo), 512-d
embeddings, cosine similarity, input normalised as (x-127.5)/127.5.

Protocol "legacy"  - reproduces the submitted manuscript: the whole 256x256 image is resized
                     to 112x112 (bilinear); no detection, no alignment.
Protocol "aligned" - recommended (R1.6d): SCRFD-10GF (buffalo_l det_10g.onnx) detects 5
                     landmarks on the GROUND-TRUTH image only (det_size 256x256, highest
                     score face). The same similarity transform (insightface norm_crop to the
                     ArcFace template, 112x112) is applied to the GT and to the output, so the
                     geometry is identical and detection never runs on generated artefacts.
                     Policy: if no face is detected on a GT image, that image is excluded from
                     the aligned ID-Sim of every method (count reported).

Rows evaluated: every method (raw and recomposited) + two references:
  Masked input  - the benchmark input I_c (FI: white holes; FU: MaskTheFace overlay)
  Visible-only (mean fill) - hole filled with the mean colour of the visible pixels.
The gain of a method over "Visible-only" is the identity signal its completion adds
beyond what the unchanged visible region already carries (R1.6e / R3.5).

Output: results/<task>/arcface_per_image.csv (long format) and arcface_gt_landmarks.json
"""
import argparse
import csv
import json

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from tqdm import tqdm

from common import DATA, METHODS, RESULTS, composite, gt_files, load_mask, load_rgb, output_path


def to_bgr8(a):
    return cv2.cvtColor((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)


class ID:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(256, 256))
        self.rec = self.app.models["recognition"]

    def emb(self, bgr112):
        f = self.rec.get_feat(bgr112).ravel()
        return f / (np.linalg.norm(f) + 1e-10)

    def legacy(self, bgr):
        return self.emb(cv2.resize(bgr, (112, 112), interpolation=cv2.INTER_LINEAR))

    def aligned(self, bgr, kps):
        return self.emb(face_align.norm_crop(bgr, landmark=kps, image_size=112))

    def detect(self, bgr):
        faces = self.app.det_model.detect(bgr, max_num=0, metric="default")
        bboxes, kpss = faces
        if bboxes is None or len(bboxes) == 0:
            return None, 0.0
        i = int(np.argmax(bboxes[:, 4]))
        return kpss[i], float(bboxes[i, 4])


def run(task, only=None):
    files = gt_files(task)
    M = ID()
    lm_p = RESULTS / task / "arcface_gt_landmarks.json"
    lm = {}
    gt_bgr, gt_e = {}, {}
    for f in tqdm(files, desc=f"{task}/GT", leave=False):
        b = to_bgr8(load_rgb(DATA[task] / "gt" / f)); gt_bgr[f] = b
        k, s = M.detect(b)
        lm[f] = {"kps": None if k is None else k.tolist(), "score": s}
        gt_e[f] = (M.legacy(b), None if k is None else M.aligned(b, k))
    json.dump(lm, open(lm_p, "w"))
    n_fail = sum(v["kps"] is None for v in lm.values())
    print(f"{task}: GT detection failures: {n_fail}/{len(files)}")
    rows = []
    entries = [(m.name, m.folder, m) for m in METHODS[task]] + [("Masked input", "_masked", None),
                                                                ("Visible-only (mean fill)", "_meanfill", None)]
    for name, folder, m in entries:
        if only and folder not in only:
            continue
        for f in tqdm(files, desc=f"{task}/{name}", leave=False):
            ic = load_rgb(DATA[task] / "masked" / f); hole = load_mask(task, f)
            if m is None:
                if folder == "_masked":
                    variants = {"raw": ic}
                else:
                    o = ic.copy(); o[hole] = ic[~hole].mean(axis=0); variants = {"raw": o}
            else:
                o = load_rgb(output_path(task, m, f, files))
                variants = {"raw": o, "comp": composite(ic, o, hole)}
            kps = lm[f]["kps"]
            for kind, img in variants.items():
                b = to_bgr8(img)
                r = {"method": name, "folder": folder, "kind": kind, "file": f,
                     "idsim_legacy": float(M.legacy(b) @ gt_e[f][0]),
                     "idsim_aligned": "" if kps is None else float(M.aligned(b, np.array(kps)) @ gt_e[f][1])}
                rows.append(r)
        sub = [r for r in rows if r["folder"] == folder]
        for kind in sorted(set(r["kind"] for r in sub)):
            a = [r["idsim_aligned"] for r in sub if r["kind"] == kind and r["idsim_aligned"] != ""]
            l = [r["idsim_legacy"] for r in sub if r["kind"] == kind]
            print(f"{task} {name:26s} {kind}: legacy {np.mean(l):.4f} aligned {np.mean(a):.4f}", flush=True)
    out = RESULTS / task / "arcface_per_image.csv"
    if only and out.exists():
        rows = [r for r in csv.DictReader(open(out)) if r["folder"] not in only] + rows
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["fi", "fu", "all"], default="all")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    for t in (["fi", "fu"] if a.task == "all" else [a.task]):
        run(t, a.only)
