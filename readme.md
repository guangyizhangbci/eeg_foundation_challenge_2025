# CBraMod-based EEG Foundation Model 

> 🚧 Work in progress

---

## 📌 Overview

CBraMod-based EEG foundation model submitted to the **NeurIPS 2025 EEG Challenge**. 1,183 participants, 8,000+ submissions on CodaBench.

---

## 🏆 Results

| Challenges | Phase | Rank |
|-----------|-------|------|
| [Challenges 1&2 ]( Results: https://www.codabench.org/competitions/9975/#/results-tab/) | Warm-up | 🥉 3rd |
| [Challenge 2 ]( Results: https://www.codabench.org/competitions/9975/#/results-tab/) | Final | 9th |

Solo Competition. 
Ranked 3rd out of 110 teams (warm-up) and 9th out of 184 teams (challenge 2, final phase) in the NeurIPS 2025 EEG Challenge.

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
The base repo: https://github.com/wjq-learning/CBraMod
### V1 - Baseline
- Depthwise Conv2d positional encoding
- Hann window + LayerNorm preprocessing
- Simple magnitude + phase spectral projection
- Shallow encoder (12 layers)

### V2 - Deeper Encoder
- Encoder deepened to 24 layers
- Upgraded to `trunc_normal` weight initialization

### V3 - Channel Self-Attention
- Replaced Conv2d PE with channel self-attention
- Explicit 51-bin rFFT spectral projection (magnitude + phase)

### V4 - Enhanced Spectral Encoding
- `EnhancedSpectralProjection` with learnable band weights (δ/θ/α/β/γ)
- Separate mag / phase / power projections
- Gated spectral fusion (`spectral_gate`)
- Learned `mask_token`
- MAD-based robust standardization

### V5 - 200 Hz Adaptation
- `EnhancedSpectralProjection` without spectral gate
- 10 attention heads in encoder
- Adapted for 200 Hz EEG data

### V6 — Temporal Positional Encoding
- Learnable `temporal_pos_encoding`
- Patch-level `LayerNorm` preprocessing
- Retains gated spectral fusion and channel self-attention from V4

---


## 🚀 Pretraining
python /pretrain/main.py [OPTIONS]

### Example

#### Pretrain with 24-layer encoder
python pretrain_main.py \
  --n_layer 24 \
  --batch_size 512

### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--n_layer` | int | `12` | Number of transformer encoder layers |
| `--batch_size` | int | `128` | Batch size |
| `--epochs` | int | `100` | Number of pretraining epochs |
| `--lr` | float | `2e-4` | Learning rate |
| `--weight_decay` | float | `5e-2` | Weight decay |
| `--clip_value` | float | `1.0` | Gradient clipping value |
| `--d_model` | int | `100` | Model embedding dimension |
| `--dim_feedforward` | int | `800` | Feedforward layer dimension |
| `--nhead` | int | `4` | Number of attention heads |
| `--seq_len` | int | `8` | Sequence length (number of patches) |
| `--mask_ratio` | float | `0.5` | Ratio of patches to mask |
| `--need_mask` | bool | `True` | Enable masked pretraining |
| `--dropout` | float | `0.1` | Dropout rate |
| `--lr_scheduler` | str | `CosineAnnealingLR` | LR scheduler (`CosineAnnealingLR`, `ExponentialLR`, `StepLR`, `MultiStepLR`, `CyclicLR`) |
| `--model_dir` | str | `...foundation_ckpt` | Directory to save pretrained checkpoints |
| `--seed` | int | `42` | Random seed |
| `--cuda` | int | `0` | CUDA device index |
| `--parallel` | bool | `False` | Enable multi-GPU training |


## ⚙️ Finetune

python /finetune/main.py [OPTIONS]

### Example

#### Fine-tune V4 with pretrained weights (recommended)
python finetune_main.py \
  --downstream_dataset Challenge-1 \
  --epochs 15 \
  --use-selected-channels \
  --use-cbramod-v4 \
  --lr 3e-5 \
  --dropout 0.4 \
  --clip_value 1 \
  --weight_decay 1e-4

#### Train V4 from scratch (no pretrained weights)
python finetune_main.py \
  --downstream_dataset Challenge-1 \
  --epochs 10 \
  --use-selected-channels \
  --use-cbramod-v4 \
  --lr 2e-4 \
  --dropout 0.6 \
  --clip_value 1 \
  --weight_decay 1e-4 \
  --use-pretrained-weights False
  
### Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--downstream_dataset` | str | `Challenge-1` | Target dataset (`Challenge-1` or `Challenge-2`) |
| `--epochs` | int | `5` | Number of training epochs |
| `--lr` | float | `1e-4` | Learning rate |
| `--weight_decay` | float | `5e-2` | Weight decay for optimizer |
| `--dropout` | float | `0.1` | Dropout rate |
| `--clip_value` | float | `1.0` | Gradient clipping value |
| `--batch_size` | int | `64` | Batch size |
| `--optimizer` | str | `AdamW` | Optimizer (`AdamW` or `SGD`) |
| `--seed` | int | `3407` | Random seed |
| `--num_workers` | int | `12` | DataLoader workers |
| `--multi_lr` | bool | `True` | Different LR for backbone vs head |
| `--frozen` | bool | `False` | Freeze backbone weights |
| `--use-pretrained-weights` | bool | `True` | Load pretrained foundation weights |
| `--use-selected-channels` | flag | `False` | Use 29 selected EEG channels instead of all 128 |
| `--use-r-5-only` | flag | `False` | Train on R5 release only |
| `--use-regression-norm` | flag | `False` | Normalize regression labels |
| `--use-200hz` | flag | `False` | Use 200 Hz model variant |
| `--use-small-model` | flag | `False` | Use small model variant |
| `--use-cbramod-v1` | flag | `False` | Use CBraMod V1 |
| `--use-cbramod-v2` | flag | `False` | Use CBraMod V2 |
| `--use-cbramod-v3` | flag | `False` | Use CBraMod V3 |
| `--use-cbramod-v4` | flag | `False` | Use CBraMod V4 |
| `--use-cbramod-v5` | flag | `False` | Use CBraMod V5 |
| `--use-cbramod-v6` | flag | `False` | Use CBraMod V6 |
| `--model_dir` | str | `...ft_ckpt` | Directory to save model checkpoints |
| `--foundation_dir` | str | `...100hz.pth` | Path to pretrained 100 Hz weights |
| `--foundation_dir_200hz` | str | `...200hz.pth` | Path to pretrained 200 Hz weights |



## 📚 Citation

If you find this repository useful, please visit my [Google Scholar](https://scholar.google.com/citations?user=ITguoVwAAAAJ&hl=en&oi=ao) and cite any relevant work, it really helps!

## 📬 Contact

If you have any questions, feel free to reach out at: patrickharvard2023@gmail.com




