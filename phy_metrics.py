"""
PhyCDNet Physics-Grounded Evaluation Metrics
=============================================
  - compute_tdi()              : Terrain Decoupling Index (IDTPD)
  - compute_rcq()              : Radiometric Compensation Quality (PCRA)
  - compute_eas()              : Edge Alignment Score (VDR)
  - compute_decision_entropy() : Mean Decision Entropy (VDR)
  - compute_mee()              : Mean Endpoint Error
  - PhysicsMetricEvaluator     : Dataset-level accumulator
"""

import torch
import torch.nn.functional as F
from typing import Dict


# ================================================================
# 核心指标函数
# ================================================================

def compute_tdi(F_ter: torch.Tensor, F_chg: torch.Tensor) -> torch.Tensor:
    """
    TDI = <F_ter, F_chg>_F / (N * C)
    Lower (near-zero) → better terrain-change decoupling.
    Eq. in Section 4.1.3.
    """
    B, C, H, W = F_ter.shape
    N = H * W
    inner = (F_ter * F_chg).sum(dim=[1, 2, 3])  # (B,)
    return (inner / (N * C)).mean()


def compute_mee(phi: torch.Tensor, affine: torch.Tensor) -> torch.Tensor:
    """
    MEE = mean(||phi - phi_gt||) 在归一化坐标下的端点误差
    phi:      CFDA 预测流场 [B, 2, H_f, W_f]
    affine:   Simulator 仿射矩阵 [B, 2, 3]
    """
    B, _, H_f, W_f = phi.shape
    device = phi.device
    # 规则网格 [-1, 1]
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H_f, device=device),
                            torch.linspace(-1, 1, W_f, device=device), indexing='ij')
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # [B,H_f,W_f,2]
    ones = torch.ones(B, H_f, W_f, 1, device=device)
    grid_h = torch.cat([grid, ones], dim=-1)  # [B,H_f,W_f,3]
    # GT 位移: affine @ grid_h
    warped = torch.bmm(affine, grid_h.view(B, -1, 3).transpose(1, 2)).transpose(1, 2)
    warped = warped.view(B, H_f, W_f, 2)
    phi_gt = warped - grid  # [B,H_f,W_f,2]
    phi_pred = phi.permute(0, 2, 3, 1)  # [B,H_f,W_f,2]
    return torch.sqrt(((phi_pred - phi_gt) ** 2).sum(dim=-1)).mean()


def compute_rcq(F1_corr: torch.Tensor, F2: torch.Tensor) -> torch.Tensor:
    """
    RCQ = (1 / (N * C)) * ||F1_corr - F2||_1
    Lower → better radiometric alignment.
    Eq. in Section 4.1.3.
    """
    B, C, H, W = F1_corr.shape
    N = H * W
    l1 = torch.abs(F1_corr - F2).sum(dim=[1, 2, 3])  # (B,)
    return (l1 / (N * C)).mean()


def compute_eas(M: torch.Tensor, gt: torch.Tensor,
                I2: torch.Tensor = None, boundary_width: int = 3) -> torch.Tensor:
    """
    EAS: Edge Alignment Score — boundary-region F1 score.
    
    Computes F1 only on pixels within `boundary_width` pixels of the
    ground-truth change contour. Higher EAS → sharper, more physically
    coherent boundaries.
    
    Eq. in Section 4.1.3.
    
    Args:
        M:  predicted change probability map [B, 1, H, W] or [B, H, W]
        gt: ground-truth binary change mask [B, 1, H, W] or [B, H, W]
        I2: (unused, reserved for future edge-guided variant)
        boundary_width: pixel distance from GT contour for boundary region
    Returns:
        scalar EAS (float tensor)
    """
    if M.dim() == 4 and M.shape[1] == 1:
        M = M.squeeze(1)
    if gt.dim() == 4 and gt.shape[1] == 1:
        gt = gt.squeeze(1)
    B, H, W = gt.shape
    device = gt.device
    eps = 1e-8

    eas_sum = 0.0
    for b in range(B):
        gt_b = gt[b].float()  # [H, W]
        # Compute distance transform from GT boundary
        # Boundary = edges of GT mask (morphological gradient)
        gt_np = gt_b.cpu().numpy()
        import numpy as np
        from scipy.ndimage import distance_transform_edt
        # Inner boundary (erosion) + outer boundary (dilation)
        from scipy.ndimage import binary_dilation, binary_erosion
        inner = binary_erosion(gt_np, iterations=1)
        outer = binary_dilation(gt_np, iterations=1)
        boundary = np.logical_xor(outer, inner).astype(np.float32)
        if boundary.sum() == 0:
            # No boundary → fall back to full-image F1
            boundary = np.ones_like(gt_np, dtype=np.float32)

        dist = distance_transform_edt(1.0 - boundary)
        boundary_mask = torch.from_numpy(dist <= boundary_width).float().to(device)

        M_b = M[b].float()
        # Binarize M at 0.5
        M_bin = (M_b > 0.5).float()
        
        tp = (M_bin * gt_b * boundary_mask).sum()
        fp = (M_bin * (1 - gt_b) * boundary_mask).sum()
        fn = ((1 - M_bin) * gt_b * boundary_mask).sum()
        
        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = 2 * prec * rec / (prec + rec + eps)
        eas_sum += f1

    return torch.tensor(eas_sum / B, device=device)


def compute_decision_entropy(M: torch.Tensor) -> torch.Tensor:
    """
    Mean Decision Entropy Ē(M) = -(1/N) Σ[ M log M + (1-M) log(1-M) ].
    
    Lower → more binary-decisive predictions (physically expected).
    Eq. (entropy) in Section 3.3.3 / Section 4.2.
    
    Args:
        M: change probability map [B, 1, H, W] or [B, H, W], values in (0,1)
    Returns:
        scalar mean entropy
    """
    if M.dim() == 4 and M.shape[1] == 1:
        M = M.squeeze(1)
    eps = 1e-8
    M_clamp = M.clamp(eps, 1.0 - eps)
    entropy = -(M_clamp * M_clamp.log() + (1 - M_clamp) * (1 - M_clamp).log())
    return entropy.mean()


# ================================================================
# 数据集级别累积评估器
# ================================================================

class PhysicsMetricEvaluator:
    """Dataset-level accumulator for physics-grounded evaluation metrics.
    
    Tracks: RCQ (PCRA), TDI (IDTPD), EAS (VDR), Decision Entropy (VDR), MEE.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self._tdi_sum = 0.0
        self._rcq_sum = 0.0
        self._mee_sum = 0.0
        self._eas_sum = 0.0
        self._entropy_sum = 0.0
        self._n_samples = 0

    @torch.no_grad()
    def update(self, F_ter, F_chg, F1_corr, F2,
               phi=None, affine=None, M=None, gt=None):
        """Accumulate physics metrics from one batch.
        
        Args:
            F_ter:  terrain-attributed features [B, C, H, W]
            F_chg:  change-attributed features [B, C, H, W]
            F1_corr: radiometrically corrected pre-event features [B, C, H, W]
            F2:     post-event reference features [B, C, H, W]
            phi:    displacement field (optional) [B, 2, H, W]
            affine: affine matrix (optional) [B, 2, 3]
            M:      final change probability map (optional) [B, 1, H, W]
            gt:     ground truth binary mask (optional) [B, 1, H, W]
        """
        B = F_ter.shape[0]
        self._tdi_sum += compute_tdi(F_ter, F_chg).item() * B
        self._rcq_sum += compute_rcq(F1_corr, F2).item() * B
        if phi is not None and affine is not None:
            try:
                self._mee_sum += compute_mee(phi, affine).item() * B
            except Exception:
                pass
        if M is not None and gt is not None:
            try:
                self._eas_sum += compute_eas(M, gt).item() * B
                self._entropy_sum += compute_decision_entropy(M).item() * B
            except Exception:
                pass
        self._n_samples += B

    def compute(self) -> Dict[str, float]:
        if self._n_samples == 0:
            return {"TDI": float("nan"), "RCQ": float("nan"),
                    "MEE": float("nan"), "EAS": float("nan"),
                    "Entropy": float("nan")}
        result = {
            "TDI": self._tdi_sum / self._n_samples,
            "RCQ": self._rcq_sum / self._n_samples,
        }
        if self._mee_sum > 0:
            result["MEE"] = self._mee_sum / self._n_samples
        if self._eas_sum > 0:
            result["EAS"] = self._eas_sum / self._n_samples
            result["Entropy"] = self._entropy_sum / self._n_samples
        return result

    def summary(self) -> str:
        r = self.compute()
        lines = [
            f"Physics Metrics ({self._n_samples} samples):",
            f"  RCQ    = {r['RCQ']:.6f}  (lower → better radiometric alignment)",
            f"  TDI    = {r['TDI']:.8e}  (lower → better terrain decoupling)",
        ]
        if 'EAS' in r:
            lines.append(f"  EAS    = {r['EAS']:.4f}  (higher → sharper boundaries)")
            lines.append(f"  Entropy= {r['Entropy']:.6f}  (lower → more decisive)")
        if 'MEE' in r:
            lines.append(f"  MEE    = {r['MEE']:.5f}  (lower → better alignment)")
        return '\n'.join(lines)
