# Unified Benchmark Protocol

## Face Inpainting (FI)
- **Dataset:** CelebA test split, 1,000 images, 256×256
- **Masks:** QD-IMD irregular masks; ratio distribution 0–10%: 7% | 10–20%: 13% | 20–30%: 60.8% | 30–40%: 15.3% | 40–50%: 3.5% | >50%: 0.4%
- **Metrics:** PSNR, SSIM, FID, LPIPS (AlexNet), L1, ArcFace ID-Sim

## Face Unmasking (FU)
- **Dataset:** CelebA test split, 500 images, 256×256
- **Masks:** MaskTheFace surgical-mask overlays
- **Metrics:** same six metrics

## ArcFace ID-Sim
InsightFace `buffalo_l` (ResNet-100, MS1MV3, 512-d embedding). ID-Sim = cosine(f(pred), f(gt)).
```bash
pip install insightface onnxruntime-gpu opencv-python-headless tqdm
python compute_arcface_similarity.py --root OUTPUTS --gt GT --out results.json
```
