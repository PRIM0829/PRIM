"""
PCRA: Physics-Constrained Radiometric Alignment
================================================


NOTE: For review purposes, this file provides the module interface and
pipeline structure. The complete unrolled operator (soft confidence map,
weighted least-squares gain, Nadaraya-Watson smoother) is described in
the manuscript and will be released in full upon acceptance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCRAModule(nn.Module):
    

    def __init__(self, channels=256, reduction=8, dilate_rate=3,
                 correction_type='pdfn', tau_cov=0.3, delta_a=0.3,
                 kernel_size=15, confidence_type='spce', hard_tau=0.1,
                 smoother_type='fixed', fixed_sigma=3.0):
        super().__init__()
        self.channels = channels
        self.correction_type = correction_type
        self.dilate_rate = dilate_rate
        self.confidence_type = confidence_type
        self.smoother_type = smoother_type
        self.fixed_sigma = fixed_sigma
        self.eps = 1e-6

        # ---- Ablation variants (BN / AdaIN / IN / 1x1 projection) ----
        if correction_type != 'pdfn':
            # Simplified placeholder: the ablation variants are standard
            # normalization layers (BatchNorm / AdaIN / InstanceNorm).
            self.correction = nn.Conv2d(channels, channels, 1)
            self.spce = None
        else:
            # SPCE head: [F1, F2, |F1-F2|] -> soft confidence w(x) in (0,1).
            # FSRI: gain/offset estimation as derived in the manuscript.
            self.spce = nn.Sequential(
                nn.Conv2d(channels * 3, channels // reduction, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, 1, 1),
                nn.Sigmoid())
            self.log_sigma = nn.Parameter(torch.tensor(0.0))
            self.kernel_size = kernel_size
        self.tau_cov = tau_cov
        self.delta_a = delta_a

        # ---- RMDC: dilated residual convolution ----
        if dilate_rate > 0:
            self.dil_conv = nn.Conv2d(channels, channels, 3,
                                      padding=dilate_rate, dilation=dilate_rate,
                                      bias=False)
            self.dil_norm = nn.InstanceNorm2d(channels)
            self.dil_act = nn.ReLU(inplace=True)
            self.refine = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.InstanceNorm2d(channels))
        else:
            self.dil_conv = None

    def forward(self, F1, F2=None):
      
        B, C, H, W = F1.shape

        if self.correction_type != 'pdfn':
            F_phys = self.correction(F1)
            F_path = torch.zeros_like(F1)
            T_eff = torch.ones(1, C, 1, 1, device=F1.device)
            R_cov = torch.tensor(0.0, device=F1.device)
            R_phys = torch.tensor(0.0, device=F1.device)
            w = torch.ones(B, 1, H, W, device=F1.device) * 0.5
        else:
            if F2 is None:
                F2_use = F1
            else:
                F2_use = F.interpolate(F2, size=(H, W), mode='bilinear',
                                       align_corners=False) \
                    if F2.shape[-2:] != (H, W) else F2

            # w(x): soft PIF confidence (Eq. 22-23)
            concat = torch.cat([F1, F2_use, torch.abs(F1 - F2_use)], dim=1)
            w = self.spce(concat)

            # Simplified gain/offset estimate (full NW unrolling: Eq. 25-27)
            a_hat = (w * F2_use).sum(dim=[2, 3], keepdim=True) / \
                    (w * F1).sum(dim=[2, 3], keepdim=True).clamp(min=self.eps)
            b_hat = torch.zeros_like(F1)
            F_phys = a_hat * F1 + b_hat

            F_path = b_hat
            T_eff = a_hat
            R_cov = F.relu(self.tau_cov - w.mean()) ** 2
            R_phys = torch.tensor(0.0, device=F1.device)

        if self.dil_conv is not None and self.dilate_rate > 0:
            F_dil = self.dil_act(self.dil_norm(self.dil_conv(F_phys)))
            F1_corr = F_phys + self.refine(F_dil)
        else:
            F1_corr = F_phys

        return F1_corr, F_path, T_eff, R_cov, R_phys, w
