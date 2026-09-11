# PRISM: Physics-Guided Causal Change Detection via Algorithm-to-Network Unrolling

This repository provides the official implementation of:

**PRISM: A Physics-Guided Causal Framework for Cross-Sensor Heterogeneous Change Detection in Remote Sensing Images**

---

> **Review release note.** This repository presents the complete architecture
> structure, pipeline wiring, loss formulation, and evaluation metrics of
> PRISM, together with the full algorithmic derivation in the manuscript.
> Module interfaces and module-level wiring are fully executable; the
> detailed internal operators of the three A2N-unrolled modules (PCRA's
> SPCE/FSRI kernel smoothing, IDTPD's descriptor/graph construction, and
> VDR's multi-scale pyramid) are described in the manuscript and will be
> made fully available upon acceptance.

---

## Overview

Cross-sensor heterogeneous change detection faces a **triple coupling problem**:

1. **Radiometric inconsistency**: cross-epoch atmospheric and transmittance drift shifts the apparent brightness of unchanged surfaces.
2. **Terrain-induced confounding**: mountain shadows and slope-aspect illumination variation fabricate false alarms indistinguishable from genuine change in feature space.
3. **Boundary uncertainty**: residual terrain confounders leave decoder logits soft and indeterminate near slope boundaries.

Existing approaches treat these as independent preprocessing stages, severing the computational graph and blocking gradient flow. We propose **PRISM**, a physics-guided framework built on an **Algorithm-to-Network (A2N) unrolling paradigm**: each differentiable module is derived directly from the closed-form update rule of its physical subproblem.

---

## Core Components

### 1️⃣ PCRA — Physics-Constrained Radiometric Alignment

**Physical subproblem**: the atmospheric radiative transfer equation (RTE), unrolled into a differentiable PIF (pseudo-invariant feature) alignment operator.

| Component | Role |
|---|---|
| **SPCE** (Soft PIF Confidence Estimator) | Learns a soft confidence map $w(\mathbf{x})$ from $[F_1, F_2, |F_1-F_2|]$, replacing the non-differentiable hard threshold $\tau$ of classical PIF |
| **FSRI** (Feature-Space Radiometric Inversion) | Weighted least-squares gain $\hat{\mathbf{a}}_c$ + Nadaraya–Watson smoother for offset $\hat{\mathbf{b}}_c(\mathbf{x})$; correction $F_1^{corr} = \hat{\mathbf{a}}\odot F_1 + \hat{\mathbf{b}}$ |
| **RMDC** | Dilated residual convolution compensating cross-resolution receptive-field mismatch (dilation matches the physical sensor ratio) |

Key design choice: the NW smoother bandwidth $\hat\sigma$ is **specified, not learned** — gradient-based learning converges to a suboptimal local minimum (Sec. 4.3.1).

### 2️⃣ IDTPD — Image-Derived Terrain Proxy Decoupling

**Physical subproblem**: terrain-illumination confounding (Minnaert/shadow models), solved DEM-free.

| Component | Role |
|---|---|
| **ITPDC** | Builds the 5-dim terrain proxy $\mathbf{D}(\mathbf{x}) = [S, A, \cos\Theta, \sin\Theta, H]$ (shadow probability, gradient anisotropy, dominant orientation, texture heterogeneity) directly from the post-event image |
| **TAGC** | SLIC superpixels + k-NN affinity graph $\mathbf{W}$ over descriptor space |
| **DRGCD** | Two-layer GCN with a split head routes terrain and change information into orthogonal subspaces: $F_{ter} \perp F_{chg}$ |

### 3️⃣ VDR — Variational Decision Refinement

**Physical subproblem**: differentiable replacement of non-differentiable CRF post-processing.

| Component | Role |
|---|---|
| **Feature-flow alignment** | Correlation estimator predicts a dense displacement field $\varphi$; $F_2$ is warped onto $F_1$'s grid with a terrain-adaptive smoothness regularizer |
| **DEF** (Decision Entropy Functional) | $\mathcal{E}(\mathbf{M}) = -\sum [M\log M + (1-M)\log(1-M)]$ drives the probability map toward binary decisiveness |
| **EGDR** (Edge-Guided Geometric Divergence Regularizer) | $\mathcal{R}_{geo}(\mathbf{M}) = \sum e^{-\mu\|\nabla I_2\|}\|\nabla \mathbf{M}\|$ aligns change boundaries with post-event image contours |

---

## Architecture

```
Input A (Pre-disaster, LR)                         Input B (Post-disaster, HR)
    │                                                       │
    ├──[Degradation Simulator]──→ I_A^simulated              │
    │         (random affine + downsampling)                 │
    └───────────────────────────────┬───────────────────────┘
                                    ▼
                          [Dual ResNet-18 Encoder]
                                    │
                     F1 ────────────┴─────────── F2
                      │                          │
                      ▼                          │
          ┌───[PCRA: SPCE + FSRI]───┐           │
          │  w(x) ← soft confidence  │           │
          │  â, b̂ ← weighted LS + NW │           │
          │  F1_corr = â⊙F1 + b̂      │           │
          └──────────┬───────────────┘           │
                     ▼                           │
          ┌───[IDTPD: ITPDC+TAGC+DRGCD]──┐      │
          │  D(x) ← terrain descriptor    │      │
          │  GCN split → F_ter ⟂ F_chg    │      │
          └──────────┬────────────────────┘      │
                     │                           │
                     ▼                           ▼
          ┌───[VDR: Feature-Flow Alignment]──┐
          │  φ = CorrEst(F_chg, F2)           │
          │  F2_aligned = Warp(F2, φ)         │
          └──────────┬───────────────────────┘
                     │
                     ▼
          ┌───[Difference Decoder]───┐
          │  P = |F_chg − F2_aligned| │
          └──────────┬───────────────┘
                     │
                     ▼
          ┌───[VDR: DEF + EGDR]───┐
          │  M ← refined probability│
          └────────────────────────┘
```

---

## Total Loss

```
L_total = L_CD + λ₁·(L_phys + R_phys) + λ₂·R_cov + λ₃·R_orth + λ₄·E(M) + λ₅·R_geo(M)
```

| Symbol | Name | Role |
|---|---|---|
| $L_{CD}$ | Change Detection Loss | Focal + Tversky Dice |
| $\lambda_1(L_{phys}{+}R_{phys})$ | Physical consistency (PCRA) | Path-radiance low-pass + transmittance regularity |
| $\lambda_2 R_{cov}$ | Coverage constraint (PCRA) | Prevent $w(\mathbf{x})$ collapse |
| $\lambda_3 R_{orth}$ | Orthogonality (IDTPD) | $F_{ter} \perp F_{chg}$ |
| $\lambda_4 \mathcal{E}(\mathbf{M})$ | Decision entropy (VDR-DEF) | Binary decisiveness |
| $\lambda_5 \mathcal{R}_{geo}(\mathbf{M})$ | Geometric divergence (VDR-EGDR) | Edge-aligned boundaries |

Weights follow a three-phase schedule (physics-dominant → transition → CD-dominant).

---

## Physics Evaluation Metrics

| Metric | Description | Direction |
|---|---|---|
| RCQ | Radiometric Compensation Quality $\|F_1^{corr}-F_2\|_1/(NC)$ | Lower |
| TDI | Terrain Decoupling Index $\langle F_{ter}, F_{chg}\rangle/(NC)$ | Lower |
| MEE | Mean Endpoint Error of displacement field $\varphi$ | Lower |
| EAS | Edge Alignment Score (boundary-region F1) | Higher |

---

## Repository Structure

```
release_prism/
├── README.md                    # 本说明文档
├── figs/                        # 彭水案例可视化（3 张大图）
│   ├── pengshui_caseA.png       #   场景 A：灾前 / 灾后 / 手工真值
│   ├── pengshui_caseB.png       #   场景 B：灾前 / 灾后 / 手工真值
│   └── pengshui_caseB_prism.png #   场景 B：灾前 / 灾后 / PRISM / 手工真值
└── models/                      # 模型代码（唯一代码目录）
    ├── __init__.py              #   公开 API 导出
    ├── networks.py              #   PRISM 主网络 + 退化仿真器 + 编码器/解码器
    ├── pcra.py                  #   PCRA：SPCE + FSRI + RMDC（含 BN/AdaIN/IN 变体）
    ├── idtpd.py                 #   IDTPD：ITPDC + TAGC + DRGCD
    ├── vdr.py                   #   VDR：特征流对齐 + DEF + EGDR（含 DANN/MMD/L2/Hist 变体）
    ├── losses.py                #   总损失 λ₁-λ₅ + 三阶段调度器
    └── phy_metrics.py           #   RCQ / TDI / MEE / EAS 物理指标
```

---

## Usage

```python
import torch
from models import PRISM

model = PRISM(feature_dim=256, n_class=2)   # PCRA(fixed σ̂=3) + IDTPD + VDR
I_A = torch.randn(1, 3, 256, 256)           # pre-event (low resolution)
I_B = torch.randn(1, 3, 256, 256)           # post-event (high resolution)

out = model(I_A, I_B)
pred = out['pred']          # change logits [1, 2, H, W]
M    = out['M']             # refined probability map [1, 1, H, W]
```

### Ablation switches

```python
model = PRISM(pcra_confidence='hard', pcra_smoother='fixed', pcra_fixed_sigma=3.0,
              idtpd_descriptors='full', idtpd_graph_type='knn_spatial',
              vdr_mu=1.0, ablation='no_pcra,no_idtpd')   # module bypass
```

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12, torchvision ≥ 0.13
- numpy, opencv-python, matplotlib, scipy

---

## Real-World Case Study: Pengshui 7·17 Event (2026)

To demonstrate practical utility beyond benchmarks, we evaluate PRISM on
the "Pengshui 7·17" geological disaster event in Chongqing, China.

### Dataset Description

| Item | Description |
|---|---|
| Sensor | GF-7 (Gaofen-7) satellite, courtesy of the China Centre for Resources Satellite Data and Application |
| Acquisition | Pre-disaster: 2026-06-06; Post-disaster: 2026-07-20 |
| Event | Geological disaster (landslide/debris flow) that occurred on 2026-07-17 in Pengshui County, Chongqing |
| Scenes | Two image pairs (Scene A, Scene B) covering the disaster region |
| Challenge | Dense vegetation, complex mountainous topography, sub-optimal imaging conditions post-disaster |
| Labels | Manually interpreted binary change references (changed / unchanged) |

Both scenes exhibit the threefold coupling problem that PRISM targets:
strong cross-epoch radiometric drift, terrain-induced illumination
variation on steep slopes, and cross-resolution geometric offsets
between the pre- and post-event acquisitions.

| Figure | Content |
|---|---|
| `figs/pengshui_caseA.png` | Scene A: Pre / Post / manual reference |
| `figs/pengshui_caseB.png` | Scene B: Pre / Post / manual reference |
| `figs/pengshui_caseB_prism.png` | Scene B: Pre / Post / PRISM / manual reference |

+  
+  ![Scene A: Pre-disaster / Post-disaster / Manual reference](figs/pengshui_caseA.png)
+  
+  ![Scene B: Pre-disaster / Post-disaster / Manual reference](figs/pengshui_caseB.png)
+  
+  ![Scene B with PRISM: Pre-disaster / Post-disaster / PRISM / Manual reference](figs/pengshui_caseB_prism.png)

PRISM preserves the topological integrity of the landslide bodies and
delineates accurate boundaries, while the comparison methods produce
fragmented detections with severe terrain-shadow false positives
(quantitative results in the manuscript, Table tab:comparison_pengshui).

---


