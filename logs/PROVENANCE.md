# Output provenance audit, 2026-11-13

Rule (authors, 2026-11-13; wording of 2026-11-14): the benchmark includes only methods evaluated with checkpoints released by, or obtained from, the original authors; no model trained or fine-tuned for the benchmark. GANMasker and M2UNet: code and checkpoints not public; outputs produced by their original authors (members of the survey team) with the models of the published papers; checkpoints available on request.

Method: for every benchmark output set, the folder in a method repository whose files are byte-identical (MD5) to the stored benchmark outputs was located (`scratchpad matchout.py`, now checked on every run by `benchmark_v2/make_configs.py`, field `stored_outputs_identity_checked`). The repository was then inspected for training artefacts (training configs with `MODE: 1`, loss logs, epoch-numbered checkpoints, tensorboard events).

| Task | Method | Stored outputs produced in | Finding | Decision |
|---|---|---|---|---|
| FI | MFCL | MFCL-Inpainting/MFCL_30 | No pretrained model released (README `torch.load("")`); checkpoints epochs 1–30 trained 2025-09/10; `test.py` loads local checkpoint, one mask for all images | excluded |
| FI | MuFIN | MuFIN/checkpoints/results | `config.yml` MODE 1, MAX_EPOCH 40; `log_inpaint.dat` epochs 1→40, 161,500 iterations (2025-09-16); released BaiduNetdisk model not used | excluded |
| FI | AttrFaceNet | AttrFaceNet/checkpoints/results | `config.yml` MODE 1, MODEL 4; `log_attr_inpaint.dat` from iteration 10 (2025-09-25/26); released BaiduNetdisk model not used | excluded |
| FI | CTSDG | CTSDG/results/celeba-CTSDG | A locally trained CTSDG (CTSDG/snapshots, 2025-09/10) exists, but its outputs (CTSDG_10/20/30) differ from the benchmark outputs. Re-running 8 images: official `celeba.pt` mean hole |diff| 3.4–8.0 grey levels vs local 300000.pt 5.4–11.7; official checkpoint downloaded 2026-04-26, outputs written afterwards | kept (official) |
| FU | Din et al. | GUMF | No official code/checkpoint; only implementation on disk is kaylode/facemask-removal (unofficial), local weights `stage2/model_99_99000.pth` (2024-11), no released weights | excluded |
| FU | MuFIN | MuFIN | Only MuFIN model on disk is the locally trained one | excluded |
| FU | DeepFillv2 | GatedConv | Official CelebA-HQ DeepFillv2 (TF) run on 030000/030001: mean hole |diff| 34.0 / 25.4 grey levels vs stored outputs | excluded |
| FU | GLCIC | GLCIC | Official release is Torch7; local `.t7` files are 0-byte placeholders; no run of the official model exists | excluded |
| FI | all other 28 | see `benchmark_v2/configs/*.json` | outputs identical to runs of official repositories with released checkpoints (or re-runs) | kept |
| FU | GANMasker, M2UNet | GANMasker, Face_Unmasking/ours | models of the published papers, run by their authors (team) | kept |

Correction found on the way: the FI "DeepFillv2" config previously pointed to `Generative-Image-Inpainting` (that repository is SCAT). The FI DeepFillv2 outputs come from the official `JiahuiYu/generative_inpainting` release `release_celeba_hq_256_deepfill_v2` (`examples/deepfillv2`, written by `test_all.py`).
