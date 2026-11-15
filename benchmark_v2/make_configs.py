"""Per-method benchmark records (Online Resource 2).

Written only from what can be established on disk:
  * the run that produced the stored outputs: the folder in a method repository whose files are byte-identical to the
    benchmark outputs (checked here by MD5 of the first two test files);
  * the repository (git remote URL) and commit of that code, and the checkpoint files present with SHA-256;
  * settings recovered from the run scripts/configs where they exist (RUN_NOTES);
  * otherwise the statement "official released checkpoint with the default inference settings of the official
    repository <URL> (commit <c>)", as agreed with the authors.
Output: configs/<task>_<folder>.json and configs/online_resource_2.csv
"""
import csv
import hashlib
import json
import subprocess
from pathlib import Path

from common import DATA, METHODS, ROOT, gt_files, has_rerun, output_path

HERE = Path(__file__).resolve().parent
CMP = ROOT.parent                                    # comparison/ (read-only)
FIR = ROOT / "FI" / "000_Related work"; NII = ROOT / "NII"; RW = CMP / "Face_Inpainting" / "RelatedWorks"; FUS = CMP / "Face_Unmasking"
# (task, folder) -> (code dir, [checkpoint globs], folder holding the identical stored outputs)
CODE = {
    ("fi", "AOT-GAN"): (RW / "AOT-GAN-for-Inpainting", ["experiments/celebahq/G*.pt"], RW / "AOT-GAN-for-Inpainting" / "AOT-GAN"),
    ("fi", "AttrFaceNet"): (RW / "AttrFaceNet", ["checkpoints/InpaintingModel_gen.pth"], RW / "AttrFaceNet" / "checkpoints" / "results"),
    ("fi", "BAT-Fill"): (RW / "BAT-Fill", ["checkpoints/celebahq_bat_pretrain/*"], RW / "BAT-Fill" / "BAT-Fill(celebA)"),
    ("fi", "CMT"): (FIR / "CMT", ["ckpt/CelebA.pth"], FIR / "CMT" / "results" / "CMT_FI"),
    ("fi", "CTSDG"): (NII / "CTSDG", ["Checkpoints/celeba.pt"], NII / "CTSDG" / "results" / "celeba-CTSDG"),
    ("fi", "CoModGAN"): (FIR / "co-mod-gan-pytorch", ["checkpoints/comod-ffhq-512/*.pth"], FIR / "co-mod-gan-pytorch" / "output" / "CoModGAN_FI" / "CoModGAN_256"),
    ("fi", "CoPaint"): (RW / "CoPaint", ["checkpoints/celeba256_250000.pt"], RW / "CoPaint" / "outputs" / "CoPaint"),
    ("fi", "CodeFormer"): (FIR / "CodeFormer", ["weights/CodeFormer/*.pth"], FIR / "CodeFormer" / "results" / "codeformer" / "CodeFormer_256"),
    ("fi", "DGTS"): (RW / "DGTS-Inpainting", ["code/test/checkpoints/celebA.pth"], RW / "DGTS-Inpainting" / "code" / "test" / "DGTS(celebA)" / "DGTS"),
    ("fi", "DeepFillv2"): (RW / "generative_inpainting", ["model_logs/release_celeba_hq_256_deepfill_v2/*"], RW / "generative_inpainting" / "examples" / "deepfillv2"),
    ("fi", "E2I"): (FIR / "E2I-inpainting", ["**/*.pth", "**/*.pt"], FIR / "E2I-inpainting" / "output" / "E2I_celeba"),
    ("fi", "EdgeConnect"): (NII / "edge-connect", ["**/*celeb*"], NII / "edge-connect" / "checkpoints" / "EdgConnect_FI"),
    ("fi", "GPEN"): (FIR / "GPEN", ["weights/GPEN-BFR-256*.pth", "weights/GPEN-BFR-512.pth"], FIR / "GPEN" / "examples" / "GPEN_FI" / "GPEN_256"),
    ("fi", "LaFIn"): (FIR / "lafin", ["checkpoints/celeba-hq/*.pth"], FIR / "lafin" / "checkpoints" / "celeba-hq" / "results" / "landmark_inpaint" / "result"),
    ("fi", "LaMa"): (NII, ["simpleLaMa.py"], NII / "LaMa_results" / "lama_celeba"),
    ("fi", "MADF"): (FIR / "Pytorch-MADF", ["checkpoints/celeba.pth"], FIR / "Pytorch-MADF" / "output" / "MADF_FI"),
    ("fi", "MADIN"): (FIR / "MADIN", ["checkpoints/CelebA.pth"], FIR / "MADIN" / "output" / "MADIN_FI"),
    ("fi", "MFCL"): (RW / "MFCL-Inpainting", ["checkpoints/Encoder-Decoder/30_net_*.pth"], RW / "MFCL-Inpainting" / "MFCL_30"),
    ("fi", "MISF"): (NII / "misf", ["**/*celeb*"], NII / "misf" / "checkpoints" / "MISF-CelebA"),
    ("fi", "MuFIN"): (RW / "MuFIN", ["checkpoints/InpaintingModel_gen.pth"], RW / "MuFIN" / "checkpoints" / "results"),
    ("fi", "NDMA"): (FIR / "NDMA_Facial_Inpainting", ["checkpoints/netG_model.pth"], FIR / "NDMA_Facial_Inpainting" / "results" / "NDMA_FI"),
    ("fi", "PEN-Net"): (FIR / "PEN-Net-for-Inpainting", ["release_model/pennet_celebahq_square256/gen_00000.pth"], FIR / "PEN-Net-for-Inpainting" / "output" / "PEN-Net_FI"),
    ("fi", "Pluralistic"): (FIR / "Pluralistic", ["checkpoints/celeba_random/latest_net_[EG].pth"], None),
    ("fi", "RePaint"): (RW / "RePaint", ["data/pretrained/celeba256_250000.pt"], RW / "RePaint" / "log" / "face_example" / "inpainted"),
    ("fi", "SCAT"): (FIR / "Generative-Image-Inpainting", ["src/checkpoints/CelebA_G*.pt"], FIR / "Generative-Image-Inpainting" / "src" / "results" / "SCAT_FI" / "SCAT"),
    ("fi", "StructureFlow"): (NII / "StructureFlow", ["**/*celeb*"], NII / "StructureFlow" / "result_images" / "StructureFlow_celeba"),
    ("fi", "TSFormer"): (FIR / "TSFormer", ["checkpoints/CelebA/*.pt"], None),
    ("fi", "VHII"): (FIR / "variable-hyperparameter-image-impainting", ["trained_models/celeba/celeba_VHII_efficient/*.pth"], None),
    ("fi", "WaveFill"): (FIR / "WaveFill", ["**/*.pth"], FIR / "WaveFill" / "results" / "WaveFill_FI"),
    ("fi", "latent-code"): (FIR / "latent-code-inpainting", ["ckpts/celeba_*.ckpt"], FIR / "latent-code-inpainting" / "output" / "latent-code-inpainting"),
    ("fi", "spa-former"): (FIR / "spa-former", ["checkpoints/celeba_hq/latest_net_G.pth"], None),
    ("fu", "CoModGAN"): (FIR / "co-mod-gan-pytorch", ["checkpoints/comod-ffhq-512/*.pth"], FIR / "co-mod-gan-pytorch" / "output" / "CoModGAN_FU" / "CoModGAN_256"),
    ("fu", "CoPaint"): (RW / "CoPaint", ["checkpoints/celeba256_250000.pt"], FUS / "00_SOTA" / "CoPaint"),
    ("fu", "DeepFillv2"): (None, [], FUS / "00_SOTA" / "GatedConv"),
    ("fu", "Din2020"): (CMP.parent.parent / "facemask-removal", ["weights/stage2/model_99_99000.pth"], FUS / "00_SOTA" / "GUMF"),
    ("fu", "EdgeConnect"): (NII / "edge-connect", ["**/*celeb*"], NII / "edge-connect" / "checkpoints" / "EdgConnect_FU"),
    ("fu", "GANMasker"): (None, [], FUS / "00_SOTA" / "GANMasker"),
    ("fu", "GLCIC"): (None, [], FUS / "00_SOTA" / "GLCIC"),
    ("fu", "M2UNet"): (None, [], FUS / "ours"),
    ("fu", "MuFIN"): (RW / "MuFIN", ["checkpoints/InpaintingModel_gen.pth"], FUS / "00_SOTA" / "MuFIN"),
    ("fu", "RePaint"): (RW / "RePaint", ["data/pretrained/celeba256_250000.pt"], FUS / "00_SOTA" / "RePaint"),
    ("fu", "SCAT"): (FIR / "Generative-Image-Inpainting", ["src/checkpoints/CelebA_G*.pt"], FUS / "00_SOTA" / "GII"),
}
# settings recovered from run scripts / configs on disk
RUN_NOTES = {
    "LaMa": "simple-lama-inpainting 0.1.2 (TorchScript Big-LaMa of the official LaMa release, trained on Places2) called by simpleLaMa.py on FI/masked with FI/mask",
    "DeepFillv2": "official TensorFlow release_celeba_hq_256_deepfill_v2 checkpoint, test.py called per image by test_all.py",
    "CTSDG": "official celeba.pt (Google-Drive release, 2021-10-18); test.py with the benchmark masks. A locally trained CTSDG model also exists (CTSDG/snapshots) but does not match the stored outputs; the official checkpoint reproduces them more closely (mean hole difference 3-8 vs 5-12 grey levels on 8 images)",
    "RePaint": "official celeba256_250000.pt; default face_example configuration of the official repository",
    "CoPaint": "official celeba256_250000.pt (RePaint CelebA-HQ diffusion model); default settings of the official repository",
    "CoModGAN": "third-party PyTorch port (zengxianyu/co-mod-gan-pytorch) with the converted official FFHQ-512 weights; run at 512 and resized to 256",
    "SCAT": "official CelebA_G0300000.pt of the SCAT repository (zzw-zjgsu/Generative-Image-Inpainting)",
}
EXCL_PROV = {  # N-04: why the stored outputs are not from the official checkpoint
    ("fi", "MFCL"): "no pretrained model was released (README test section has empty checkpoint paths); outputs identical to MFCL_30, written by a model trained by us for 30 epochs (checkpoints 2025-09/10)",
    ("fi", "AttrFaceNet"): "outputs identical to checkpoints/results of a model trained by us (config MODE 1, MAX_EPOCH 40, log_attr_inpaint.dat 2025-09-25/26); the released BaiduNetdisk checkpoint was not used",
    ("fi", "MuFIN"): "outputs identical to checkpoints/results of a model trained by us (config MODE 1, 40 epochs, 161,500 iterations, log_inpaint.dat 2025-09-16); the released BaiduNetdisk checkpoint was not used",
    ("fu", "MuFIN"): "the only MuFIN model in the workspace is the one trained by us; outputs cannot come from the released checkpoint",
    ("fu", "Din2020"): "no official code or checkpoint was released; outputs from a re-implementation (kaylode/facemask-removal) trained by us (weights stage2/model_99_99000.pth, 2024-11)",
    ("fu", "DeepFillv2"): "outputs not reproduced by the official CelebA-HQ DeepFillv2 checkpoint (mean hole difference 25-34 grey levels on 2 images); model of unknown provenance",
    ("fu", "GLCIC"): "outputs cannot be traced to the official Torch7 release (no run of it exists; the local .t7 files are empty placeholders)",
}


def sha(p, algo="sha256"):
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def git(d, *args):
    try:
        return subprocess.run(["git", "-C", str(d), *args], capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def rel(p):
    try:
        return str(p.relative_to(CMP))
    except ValueError:
        return str(p)


def main():
    out = HERE / "configs"; out.mkdir(exist_ok=True)
    rows = []
    for task, ms in METHODS.items():
        files = gt_files(task)[:2]
        for m in ms:
            code, pats, src = CODE.get((task, m.folder), (None, [], None))
            ck = []
            if code and code.exists():
                for pat in pats:
                    for f in sorted(code.glob(pat)):
                        if f.is_file() and f.stat().st_size > 1e5:
                            ck.append({"file": rel(f), "bytes": f.stat().st_size,
                                       "sha256": sha(f) if f.stat().st_size <= 2.5e9 else "not hashed (>2.5 GB)"})
            url = git(code, "remote", "get-url", "origin") if code and (code / ".git").exists() else None
            commit = git(code, "rev-parse", "HEAD") if code and (code / ".git").exists() else None
            src_ok = None
            if src is not None and src.exists():
                src_ok = all((src / f).exists() and sha(src / f, "md5") == sha(DATA[task] / "SOTA" / m.folder / f, "md5") for f in files)
            rerun = has_rerun(task, m)
            if (task, m.folder) in EXCL_PROV:
                settings = "EXCLUDED: " + EXCL_PROV[(task, m.folder)]
            elif m.team:
                settings = "outputs produced by the method's authors (members of this team) with the model of the published paper"
            elif rerun:
                settings = "re-run with the official checkpoint and the benchmark masks (rerun_command)"
            elif m.folder in RUN_NOTES or m.name in RUN_NOTES:
                settings = RUN_NOTES.get(m.folder) or RUN_NOTES[m.name]
            else:
                settings = ("official released checkpoint with the default inference settings of the official repository"
                            + (f" {url}" if url else "") + (f" (commit {commit[:7]})" if commit else ""))
            rec = {"task": task, "method": m.name, "citation_key": m.cite, "category": m.category, "family": m.family,
                   "status": m.status, "note": m.note, "developed_by_author_team": m.team,
                   "input_resolution": "256x256 (benchmark); inputs of other sizes resized with Lanczos and outputs resized back",
                   "repository": url, "git_commit": commit, "local_code": rel(code) if code and code.exists() else None,
                   "checkpoints_present": ck,
                   "stored_outputs_produced_in": rel(src) if src is not None else ("benchmark_v2/reruns" if rerun else None),
                   "stored_outputs_identity_checked": src_ok,
                   "rerun": rerun, "rerun_command": {"spa-former": "python rerun_code/run_spaformer.py", "VHII": "python rerun_code/run_vhii.py",
                                                                  "Pluralistic": "python rerun_code/run_pluralistic_seeds.py --task fi --subset all --k 1"}.get(m.folder),
                   "run_settings": settings, "stochastic": m.stochastic,
                   "evaluation": "recomposited with Eq. (2); metrics per benchmark_v2/eval_pixel.py, eval_fid.py, eval_arcface.py"}
            json.dump(rec, open(out / f"{task}_{m.folder}.json", "w"), indent=1)
            rows.append({"task": task, "method": m.name, "category": m.category, "status": m.status, "rerun": rerun,
                         "repository": url or "", "git_commit": commit or "", "outputs_produced_in": rec["stored_outputs_produced_in"] or "",
                         "outputs_identity_checked": src_ok, "checkpoints": "; ".join(f"{c['file']} [{c['sha256'][:12]}]" for c in ck),
                         "run_settings": settings})
            print(task, m.name, "| src ok:", src_ok, "|", (url or "-"), (commit or "")[:7], "|", len(ck), "ckpt")
    with open(out / "online_resource_2.csv", "w", newline="", encoding="utf8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
