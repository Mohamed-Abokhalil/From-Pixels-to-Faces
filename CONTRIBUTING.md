# Contributing

We welcome additions of new face inpainting / unmasking papers, code links, and benchmark results.

**To add a paper**, open a PR editing the relevant file in `papers/` using the row format:
```
| Year | Method | Venue | Backbone | Code | Paper |
```
Please include an official code link when available and keep entries in chronological order.

**To report benchmark results**, follow the protocol in `benchmark/PROTOCOL.md` (CelebA test split, QD-IMD or MaskTheFace masks, 256×256) and attach the metric JSON produced by `compute_arcface_similarity.py`.
