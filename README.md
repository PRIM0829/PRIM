
**PRISM: A Physics-Guided Causal Framework for Cross-Sensor Heterogeneous Change Detection in Remote Sensing Images**



---


## Total Loss

```
L_total = L_CD + λ₁·(L_phys + R_phys) + λ₂·R_cov + λ₃·R_orth + λ₄·E(M) + λ₅·R_geo(M)
```

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
├── README.md
└── models/
    ├── __init__.py          # Public API
    ├── networks.py          # PRISM, DegradationSimulator, encoder/decoder
    ├── pcra.py              # PCRA: SPCE + FSRI + RMDC (with BN/AdaIN/IN variants)
    ├── idtpd.py             # IDTPD: ITPDC + TAGC + DRGCD
    ├── vdr.py               # VDR: feature-flow + DEF + EGDR (with DANN/MMD/L2/Hist variants)
    ├── losses.py            # Total loss, 3-phase scheduler
    └── phy_metrics.py       # RCQ, TDI, MEE, EAS
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

