# 🧠 CBraMod — NeurIPS 2025 EEG Challenge

> 🚧 Work in progress

---

## 📌 Overview

CBraMod-based EEG foundation model submitted to the **NeurIPS 2025 EEG Challenge** — 1,183 teams, 8,000+ submissions on CodaBench.

---

## 🏆 Results

| Challenge | Phase | Rank |
|-----------|-------|------|
| [Challenges 1&2 ](https://eeg2025.github.io/leaderboard/) | Warm-up | 🥉 3rd |
| [Challenge 2](https://eeg2025.github.io/leaderboard/) | Final | 9th |

> ⚠️ GPU-limited submission — V5/V6 and ensemble strategies were not fully explored.

---

## 🗂️ Repository Structure

```
├── models/
│   ├── cbramod_v1.py
│   ├── cbramod_v2.py
│   ├── cbramod_v3.py
│   ├── cbramod_v4.py
│   ├── cbramod_v5.py
│   └── cbramod_v6.py
├── datasets/
├── pretrain_main.py
├── pretrain_trainer.py
├── finetune_main.py
├── finetune_trainer_multir.py
└── finetune_evaluator.py
```

---

## 🔬 Model Versions

### V1 — Baseline
- Depthwise Conv2d positional encoding
- Hann window + LayerNorm preprocessing
- Simple magnitude + phase spectral projection
- Shallow encoder (12 layers)

### V2 — Deeper Encoder
- Encoder deepened to 24 layers
- Upgraded to `trunc_normal` weight initialization

### V3 — Channel Self-Attention
- Replaced Conv2d PE with channel self-attention
- Explicit 51-bin rFFT spectral projection (magnitude + phase)

### V4 — Enhanced Spectral Encoding
- `EnhancedSpectralProjection` with learnable band weights (δ/θ/α/β/γ)
- Separate mag / phase / power projections
- Gated spectral fusion (`spectral_gate`)
- Learned `mask_token`
- MAD-based robust standardization

### V5 — 200 Hz Adaptation
- `EnhancedSpectralProjection` without spectral gate
- 10 attention heads in encoder
- Adapted for 200 Hz EEG data

### V6 — Temporal Positional Encoding
- Learnable `temporal_pos_encoding`
- Patch-level `LayerNorm` preprocessing
- Retains gated spectral fusion and channel self-attention from V4

---

## 📚 Citation

If you find this repository useful, please visit my [Google Scholar](https://scholar.google.com/citations?user=ITguoVwAAAAJ&hl=en&oi=ao) and cite any relevant work — it really helps!
