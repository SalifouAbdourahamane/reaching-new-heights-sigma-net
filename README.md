# SIGMA-Net: Satellite-grounded Inference for Geospatial Map Anchoring

**1st place solution** — ESA Φ-lab × ITU × AI for Good: *Reaching New Heights with GeoFM* Challenge  
**Team**: The_Alchemist &nbsp;·&nbsp; **Public leaderboard score: 0.6030**

| IoU Building | IoU Vegetation | IoU Water | RMSE Bld Height | RMSE Veg Height | **Final Score** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.6277 | 0.9193 | 0.7129 | 1.608 m | 2.868 m | **0.6030** |

---

## Architecture

![SIGMA-Net Architecture](Sigma_net.png)

---

## Core idea: Master Problem vs Slave Problem

The competition requires predicting five quantities per 256×256 tile from compressed 64-channel AlphaEarth satellite embeddings: building cover, vegetation cover, water cover, building height, and vegetation height. Solved naively — a single model trained end-to-end from embedding to output — this is the *master problem*. It is hard because the model must simultaneously internalize geographic structure, spectral-to-semantic mappings, object boundary priors, and cross-task correlations across a heterogeneous loss surface.

Our key observation is that Stage 1 — a multitask ConvNeXt UNet we trained end-to-end on the competition data — already produces strong predictions for vegetation and water, but carries systematic local errors on buildings and height in complex urban scenes. The residual between this prior and the ground truth is far smaller and more structured than the full prediction space.

We therefore define the *slave problem*: **given a mostly-correct prediction map, identify precisely where it is wrong and estimate the correction**. This is structurally easier to learn:

| | Master Problem | Slave Problem (SIGMA-Net) |
|---|---|---|
| **Output space** | Unbounded per-pixel prediction | Bounded residual: ΔCover ∈ [−0.05, +0.15], Δh ∈ [−2, +5] m |
| **What must be learned** | Geography + spectral mapping + structural priors | Systematic error patterns of a known prior |
| **Model capacity needed** | Large pretrained GFM | 9.2M-parameter corrector |
| **Supervision signal** | Dense, high-dimensional | Structured: binary error masks + signed residuals |

---

## Stage 1 — GeoMTConvNeXt (Anchor Predictions)

The first stage is **GeoMTConvNeXt**, a multitask ConvNeXt UNet we trained from scratch on the embed2heights training set. The model takes the 64-channel AlphaEarth embedding as input and is optimised jointly on all five competition tasks (building cover, vegetation cover, water cover, building height, vegetation height) using a combination of L1, Lovász IoU, and masked SmoothL1 losses. Its weights are hosted at [`Abdoul27/embed2heights-geoconvnext`](https://huggingface.co/Abdoul27/embed2heights-geoconvnext).

GeoMTConvNeXt produces a **4-channel prior map** `[4, 256, 256]` per tile:

| Channel | Content | Range |
|---------|---------|-------|
| ch 0 | Building cover probability | [0, 1] |
| ch 1 | Vegetation cover probability | [0, 1] |
| ch 2 | Water cover probability | [0, 1] |
| ch 3 | Building / vegetation height | metres |

**Implementation note**: predictions are pre-computed and stored in an indexed `.npz` file on HuggingFace, keyed by a normalised fingerprint of the AlphaEarth embedding. At inference time, Stage 1 is a fast dictionary lookup — no GPU forward pass required.

---

## Stage 2 — SIGMA-Net (Confidence-gated Residual Corrector)

SIGMA-Net operationalises the slave problem through a structured three-question decomposition applied independently per task:

> 1. **Is the prior wrong here?** → *Discrepancy Localizer* (binary spatial detection)
> 2. **If wrong, in which direction and by how much?** → *Residual Estimator* (bounded regression)
> 3. **How confident are we in this correction?** → *Correction Confidence* (calibrated probability)

### Input

Both the AlphaEarth embedding (64 channels) and the Stage 1 prior map (4 channels) are concatenated at the input, forming a **68-channel joint representation**. This gives SIGMA-Net full access to both the raw satellite signal and the current prediction it must improve — the network can compare what the satellite "sees" with what the prior "claims" and resolve the discrepancy.

### Backbone

The joint input is processed by a **4-level ConvNeXt UNet encoder–decoder**. Each encoder block applies depthwise convolution (7×7 kernel, grouped), BatchNorm, GELU activation, and a 4× pointwise channel expansion. Skip connections from each encoder level are concatenated with the corresponding decoder level, preserving fine spatial detail critical for boundary-accurate segmentation.

### Six Prediction Heads

Six isolated heads branch from the final decoder feature map (64 channels):

| Head | Output | Activation | Role |
|------|--------|-----------|------|
| `discrepancy_localizer` | [B, 3, H, W] | Sigmoid | Per-pixel error detection for cover (building, veg, water) |
| `residual_estimator` | [B, 3, H, W] | Tanh | Signed correction direction ∈ [−1, +1] |
| `correction_confidence` | [B, 3, H, W] | Sigmoid | Reliability of the proposed cover correction |
| `height_discrepancy_localizer` | [B, 1, H, W] | Sigmoid | Per-pixel error detection for height |
| `height_residual_estimator` | [B, 1, H, W] | Tanh | Signed height correction direction |
| `height_correction_confidence` | [B, 1, H, W] | Sigmoid | Reliability of the proposed height correction |

### Evidence Gate and Correction Fusion

Corrections propagate through a **triple evidence gate** that requires the discrepancy detector and confidence head to jointly agree before any correction reaches the output:

```
# Cover (per channel c):
gate_c    = discrepancy_prob[:, c]  ×  correction_confidence[:, c]
Δcover    = clamp(gate_c × residual[:, c] × 0.15,  −0.05, +0.15)
cover_out = clamp(prior_cover[:, c] + Δcover,        0.0,   1.0)

# Height:
gate_h    = height_discrepancy_prob  ×  height_correction_confidence
Δheight   = clamp(gate_h × height_residual × 5.0,  −2.0, +5.0)
height_out= clamp(prior_height + Δheight,            0.0,  60.0)
```

The **asymmetric cover clamp** [−0.05, +0.15] and **asymmetric height clamp** [−2, +5] m encode an empirical observation: the Foundation Model under-predicts both coverage fractions and heights in dense urban areas more often than it over-predicts them. The gate design ensures SIGMA-Net is *conservative by default* — in regions where the discrepancy and confidence heads do not agree, the prior map passes through unchanged.

---

## Training

All six head supervision targets are **derived automatically** from the pixel-wise residual between the Stage 1 prior and the ground-truth label — no additional annotation is required.

| Loss term | Formula | Purpose |
|-----------|---------|---------|
| `L_cover` | L1 + Lovász IoU surrogate | Directly optimises the IoU competition metric |
| `L_discrepancy` | BCE | Trains the localizer to detect prior errors |
| `L_residual` | MSE masked to error regions | Trains the estimator only where corrections are needed |
| `L_confidence` | BCE | Calibrates the confidence output |
| `L_tv` | Total variation on residuals | Spatial smoothness — prevents noisy correction fields |
| `L_height` | Masked SmoothL1 (building ×2, veg ×1.5) | Height with class-weighted emphasis on buildings |

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Scheduler | Cosine annealing, η_min = 10⁻⁶ |
| Epochs | 40 |
| Batch size | 8 |
| Mixed precision | bfloat16 autocast |
| Parameters | 9.2M |

---

## Reproduce

### 1. Get the dataset

The competition dataset is hosted on [EOTDL](https://www.eotdl.com):

```bash
pip install eotdl
eotdl auth login
eotdl datasets get embed2heights
```

Expected layout after extraction:

```
embed2heights/data/
  train/
    alphaearth_emb/        # gee_emb_<core>.tif        (2024 training tiles)
    labels/                # label_<core>_<year>.tif
  test/
    alphaearth_test_emb/   # emb_<core>_quantized.tif  (946 test tiles)
```

> **Note for organizers**: if the dataset is already extracted, set `DATA_ROOT` in Section 2 of the notebook to point to the `data/` subfolder.

### 2. Install and run

```bash
pip install -r requirements.txt
jupyter notebook sigma_net_reproduce.ipynb
```

In **Section 2**, set:

```python
INFERENCE_ONLY = True                     # use pretrained weights (default, ~5 min)
DATA_ROOT = Path('/path/to/embed2heights/data')
```

Run all cells. The notebook downloads Stage 1 anchor index + SIGMA-Net weights from HuggingFace, runs inference on all 946 test tiles, and writes `outputs/sigma_net_submission.zip`.

To **train from scratch** (~60 min on A40), set `INFERENCE_ONLY = False`.

---

## HuggingFace assets

All pretrained weights are at [`Abdoul27/embed2heights-geoconvnext`](https://huggingface.co/Abdoul27/embed2heights-geoconvnext):

| File | Size | Description |
|------|------|-------------|
| `predictions.npz` | 333 MB | Stage 1 anchor index (fingerprint → 4-channel prediction for all tiles) |
| `norm_stats.npz` | 29 KB | AlphaEarth per-channel normalisation statistics |
| `sigma_net_best.pt` | 36 MB | SIGMA-Net weights (epoch 27, val score 0.5588 → LB 0.6030) |
