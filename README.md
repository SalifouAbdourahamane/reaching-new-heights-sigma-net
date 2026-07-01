# SIGMA-Net: Satellite-grounded Inference for Geospatial Map Anchoring

**1st place solution** — ESA Φ-lab × ITU × AI for Good: *Reaching New Heights with GeoFM* Challenge  
**Team**: The_Alchemist &nbsp;·&nbsp; **Public leaderboard score: 0.6030**

**Author**: Abdourahamane Ide Salifou &nbsp;·&nbsp; [aidesali@andrew.cmu.edu](mailto:aidesali@andrew.cmu.edu)  
**Affiliation**: Carnegie Mellon University Africa — Engineering Student in Artificial Intelligence

| IoU Building | IoU Vegetation | IoU Water | RMSE Bld Height | RMSE Veg Height | **Final Score** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.6277 | 0.9193 | 0.7129 | 1.608 m | 2.868 m | **0.6030** |

---

## Architecture

![SIGMA-Net Architecture](Sigma_net.png)

---

## Abstract

We present SIGMA-Net, a two-stage residual correction framework for simultaneous building/vegetation cover segmentation and height estimation from AlphaEarth satellite embeddings. The core insight is a deliberate decomposition of the prediction problem: rather than training a single model to solve the full four-task prediction problem from scratch (the *master problem*), we first train a strong multitask model — GeoMTConvNeXt — to generate a structural prior, and then train a compact specialist network whose sole objective is to detect and correct the errors in that prior (the *slave problem*). This decomposition reduces the effective output search space by an order of magnitude, enables fast convergence with limited training data, and produces a final model of only 9.2 million parameters that achieves competitive results on all five competition metrics.

---

## 1. Problem Framing: Master vs. Slave Problem

The competition requires predicting five quantities per 256×256 tile from compressed 64-channel AlphaEarth satellite embeddings: building cover, vegetation cover, water cover, building height, and vegetation height. Solved naively — a single model trained end-to-end from embedding to output — this is the *master problem*. It is hard because the model must simultaneously internalize geographic structure, spectral-to-semantic mappings, object boundary priors, and cross-task correlations across a heterogeneous loss surface.

Our key observation is that Stage 1 — GeoMTConvNeXt, a multitask model we trained end-to-end on the competition data — already produces predictions that are a **credible but imperfect prior**: they capture global structure and achieve strong performance on vegetation and water, but carry systematic local errors on buildings and height in complex urban scenes. The residual between this prior and the ground truth is far smaller and more structured than the full prediction space.

We therefore define the *slave problem*: **given a mostly-correct prediction map, identify precisely where it is wrong and estimate the correction**. This is structurally easier to learn:

| | Master Problem | Slave Problem (SIGMA-Net) |
|---|---|---|
| **Output space** | Unbounded per-pixel prediction | Bounded residual: ΔCover ∈ [−0.05, +0.15], Δh ∈ [−2, +5] m |
| **What must be learned** | Geography + spectral mapping + structural priors | Systematic error patterns of a known prior |
| **Model capacity needed** | Large end-to-end model | 9.2M-parameter corrector |
| **Supervision signal** | Dense, high-dimensional | Structured: binary error masks + signed residuals |

---

## 2. Stage 1 — GeoMTConvNeXt Prior

The first stage uses **GeoMTConvNeXt**, a ConvNeXt-based Geospatial Multitask model (hosted at [`Abdoul27/embed2heights-geoconvnext`](https://huggingface.co/Abdoul27/embed2heights-geoconvnext)). This model is not a generic backbone: it was trained end-to-end on the joint five-task objective over the full embed2heights training set, using the 64-channel AlphaEarth/Tessera embeddings provided by the competition. These embeddings are dense compressed representations of multi-temporal, multi-spectral Earth Observation imagery, encoding rich spectral and temporal context into a 64-dimensional feature volume per spatial location.

GeoMTConvNeXt was trained end-to-end on the joint five-task objective over the full training set. Its weights encode: (i) global land-cover semantics extracted from AlphaEarth EO data, (ii) task-specific decision boundaries calibrated to the competition label schema, and (iii) the structural biases of satellite-derived height fields across European urban and peri-urban landscapes.

The model produces a **4-channel prior map** `[4, 256, 256]` per tile:

| Channel | Content | Range |
|---------|---------|-------|
| ch 0 | Building cover probability | [0, 1] |
| ch 1 | Vegetation cover probability | [0, 1] |
| ch 2 | Water cover probability | [0, 1] |
| ch 3 | Building / vegetation height | metres |

**Implementation note**: predictions are pre-computed and stored in an indexed `.npz` file on HuggingFace, keyed by a normalised fingerprint of the AlphaEarth embedding. At inference time, Stage 1 is a fast dictionary lookup — no GPU forward pass is required during submission generation.

---

## 3. Stage 2 — SIGMA-Net: Confidence-gated Residual Corrector

SIGMA-Net operationalises the slave problem through a structured three-question decomposition applied independently per task:

> 1. **Is the prior wrong here?** → *Discrepancy Localizer* (binary spatial detection)
> 2. **If wrong, in which direction and by how much?** → *Residual Estimator* (bounded regression)
> 3. **How confident are we in this correction?** → *Correction Confidence* (calibrated probability)

### 3.1 Input Representation

Both the AlphaEarth embedding (64 channels) and the Stage 1 prior map (4 channels) are concatenated at the input, forming a **68-channel joint representation**. This gives SIGMA-Net full access to both the raw satellite signal and the current prediction it must improve — the network can compare what the satellite "sees" with what the prior "claims" and resolve the discrepancy.

### 3.2 Backbone: ConvNeXt UNet

The joint input is processed by a **4-level ConvNeXt UNet encoder–decoder**. Each encoder block applies depthwise convolution (7×7 kernel, grouped), BatchNorm, GELU activation, and a 4× pointwise channel expansion — a computationally efficient pattern that captures both local texture and multi-scale structural context. Skip connections from each encoder level are concatenated with the corresponding decoder level, preserving fine spatial detail critical for boundary-accurate segmentation.

### 3.3 Six Prediction Heads

Six isolated prediction heads branch from the final decoder feature map (64 channels):

| Head | Output | Activation | Role |
|------|--------|-----------|------|
| `discrepancy_localizer` | [B, 3, H, W] | Sigmoid | Per-pixel error detection for cover (building, veg, water) |
| `residual_estimator` | [B, 3, H, W] | Tanh | Signed correction direction ∈ [−1, +1] |
| `correction_confidence` | [B, 3, H, W] | Sigmoid | Reliability of the proposed cover correction |
| `height_discrepancy_localizer` | [B, 1, H, W] | Sigmoid | Per-pixel error detection for height |
| `height_residual_estimator` | [B, 1, H, W] | Tanh | Signed height correction direction |
| `height_correction_confidence` | [B, 1, H, W] | Sigmoid | Reliability of the proposed height correction |

### 3.4 Evidence Gate and Correction Fusion

Corrections are applied through a **triple evidence gate** that requires the discrepancy detector and confidence head to jointly agree before any correction propagates to the output:

```
# Cover (per channel c):
gate_c    = discrepancy_prob[:, c]  ×  correction_confidence[:, c]
Δcover    = clamp(gate_c × residual[:, c] × 0.15,  −0.05, +0.15)
cover_out = clamp(prior_cover[:, c] + Δcover,        0.0,   1.0)

# Height:
gate_h     = height_discrepancy_prob  ×  height_correction_confidence
Δheight    = clamp(gate_h × height_residual × 5.0,  −2.0, +5.0)
height_out = clamp(prior_height + Δheight,            0.0,  60.0)
```

The **asymmetric cover clamp** [−0.05, +0.15] and **asymmetric height clamp** [−2, +5] m encode an empirically observed asymmetry: Stage 1 under-predicts both coverage fractions and heights in dense urban areas more often than it over-predicts them. The gate design ensures SIGMA-Net is *conservative by default* — in regions where the discrepancy and confidence heads do not agree, the prior map passes through unchanged.

---

## 4. Training

All six head supervision targets are **derived automatically** from the pixel-wise residual between the Stage 1 prior and the ground-truth label — no additional annotation is required:

- **Discrepancy targets**: binary masks where `|prior − GT| > 0.10` (cover) or `> 1.0 m` (height)
- **Residual targets**: signed error normalised to [−1, +1] by the correction scale (0.15 for cover, 5.0 m for height)
- **Confidence targets**: softer binary masks at half the discrepancy threshold

| Loss term | Formula | Purpose |
|-----------|---------|---------|
| `L_cover` | L1 + Lovász IoU surrogate | Directly optimises the IoU competition metric |
| `L_discrepancy` | BCE | Trains the localizer to detect prior errors |
| `L_residual` | MSE masked to error regions only | Trains the estimator only where corrections are needed |
| `L_confidence` | BCE | Calibrates the confidence output |
| `L_tv` | Total variation on residuals | Spatial smoothness — prevents noisy correction fields |
| `L_height` | Masked SmoothL1 (building ×2, veg ×1.5) + head BCE/MSE | Height with class-weighted emphasis on buildings |

The **Lovász loss** is critical for the building and water cover tasks: it directly optimises IoU as a differentiable convex surrogate, whereas standard BCE is insensitive to rare foreground pixels in highly imbalanced tiles.

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| Scheduler | Cosine annealing, η_min = 10⁻⁶ |
| Epochs | 40 |
| Batch size | 8 |
| Mixed precision | bfloat16 autocast |
| Validation split | 20% random hold-out |
| Checkpoint gate | Building IoU > 0.35 |

Convergence is fast: the best checkpoint is typically reached within 10–15 epochs, a direct consequence of the compact residual output space.

---

## 5. Reproduce

### Get the dataset

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

### Run

```bash
pip install -r requirements.txt
jupyter notebook sigma_net_reproduce.ipynb
```

In **Section 2** of the notebook:

```python
INFERENCE_ONLY = True                          # ~5 min: download pretrained weights and run inference
DATA_ROOT = Path('/path/to/embed2heights/data')
```

Run all cells. The notebook downloads the Stage 1 anchor index and SIGMA-Net weights from HuggingFace, runs inference on all 946 test tiles, and writes `outputs/sigma_net_submission.zip`.

To **train from scratch** (~60 min on A40), set `INFERENCE_ONLY = False`.

---

## HuggingFace assets

[`Abdoul27/embed2heights-geoconvnext`](https://huggingface.co/Abdoul27/embed2heights-geoconvnext)

| File | Size | Description |
|------|------|-------------|
| `predictions.npz` | 333 MB | Stage 1 anchor index (fingerprint → [4, 256, 256] prediction) |
| `norm_stats.npz` | 29 KB | AlphaEarth per-channel normalisation statistics |
| `sigma_net_best.pt` | 36 MB | SIGMA-Net weights (epoch 27 → LB 0.6030) |
