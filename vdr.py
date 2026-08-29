"""
VDR: Variational Decision Refinement
====================================

NOTE: For review purposes, this file provides the module interface and
pipeline structure. The complete differentiable energy formulation is
described in the manuscript and will be released in full upon acceptance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# DEF: Decision Entropy Functional
# ================================================================
def compute_def_loss(M):

    eps = 1e-8
    M_c = M.clamp(eps, 1 - eps)
    entropy = -(M_c * torch.log(M_c) + (1 - M_c) * torch.log(1 - M_c))
    return entropy.mean()


# ================================================================
# EGDR: Edge-Guided Geometric Divergence Regularizer
# ================================================================
def compute_egdr_loss(M, I2, mu=1.0):
    
    B, _, H, W = M.shape
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=torch.float32, device=M.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)
    Mx = F.conv2d(F.pad(M, (1, 1, 1, 1), mode='reflect'), sobel_x)
    My = F.conv2d(F.pad(M, (1, 1, 1, 1), mode='reflect'), sobel_y)
    grad_M = torch.sqrt(Mx * Mx + My * My + 1e-8)
    if I2.shape[-2:] != (H, W):
        I2 = F.interpolate(I2, size=(H, W), mode='bilinear', align_corners=False)
    I2_gray = I2.mean(dim=1, keepdim=True)
    I2x = F.conv2d(F.pad(I2_gray, (1, 1, 1, 1), mode='reflect'), sobel_x)
    I2y = F.conv2d(F.pad(I2_gray, (1, 1, 1, 1), mode='reflect'), sobel_y)
    grad_I2 = torch.sqrt(I2x * I2x + I2y * I2y + 1e-8)
    weight = torch.exp(-mu * grad_I2)
    return (weight * grad_M).mean()


# ================================================================
# CorrelationEstimator: dense displacement field φ
# ================================================================
class CorrelationEstimator(nn.Module):


    def __init__(self, channels=256):
        super().__init__()
        self.coarse_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels), nn.ReLU(inplace=True))
        self.refine_conv1 = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels // 2), nn.ReLU(inplace=True))
        self.refine_conv2 = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 4, 3, padding=1, bias=False),
            nn.InstanceNorm2d(channels // 4), nn.ReLU(inplace=True))
        self.flow_head = nn.Sequential(
            nn.Conv2d(channels // 4, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 3, padding=1), nn.Tanh())

    def forward(self, F_chg, F2):
        H, W = F_chg.shape[-2:]
        F2_r = F.interpolate(F2, size=(H, W), mode='bilinear', align_corners=False)
        concat = torch.cat([F_chg, F2_r], dim=1)
        coarse = self.coarse_conv(concat)
        r1 = self.refine_conv1(coarse)
        r2 = self.refine_conv2(r1)
        return self.flow_head(r2)


# ================================================================
# VDRModule: feature alignment (Stage 1) + DEF/EGDR (Stage 2)
# ================================================================
class VDRModule(nn.Module):


    def __init__(self, channels=256, alpha=1e-3,
                 alignment_type='cfda',
                 pyramid_levels=3,
                 temperature=0.07,
                 smoothness_type='terrain',
                 smoothness_kernel=17,
                 mu_egdr=1.0,
                 mu_learnable=True):
        super().__init__()
        self.channels = channels
        self.alpha = alpha
        self.alignment_type = alignment_type
        if mu_learnable:
            self.mu_egdr = nn.Parameter(torch.tensor(float(mu_egdr)))
        else:
            self.register_buffer('mu_egdr', torch.tensor(float(mu_egdr)))
        if alignment_type == 'cfda':
            self.corr_estimator = CorrelationEstimator(channels=channels)
        elif alignment_type != 'none':
            # Ablation variants (MMD / DANN / ℓ2 / histogram) follow
            # the manuscript; simplified identity is used here.
            self.alignment = None
        else:
            self.corr_estimator = None
        self.smoothness_type = smoothness_type
        self.smoothness_kernel = smoothness_kernel

    def _warp(self, F2, phi):
        B, C, h_out, w_out = phi.shape[0], F2.shape[1], phi.shape[2], phi.shape[3]
        theta = torch.eye(2, 3, device=phi.device).unsqueeze(0).repeat(B, 1, 1)
        base_grid = F.affine_grid(theta, (B, 1, h_out, w_out), align_corners=False)
        sample_grid = base_grid + phi.permute(0, 2, 3, 1)
        return F.grid_sample(F2, sample_grid, mode='bilinear',
                             padding_mode='border', align_corners=False)

    def _smooth_loss(self, phi, dem_gradient=None):
        B, _, H, W = phi.shape
        dx = phi[:, :, :, 1:] - phi[:, :, :, :-1]
        dy = phi[:, :, 1:, :] - phi[:, :, :-1, :]
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        gm = (dx ** 2 + dy ** 2).sum(dim=1, keepdim=True)
        if self.smoothness_type == 'isotropic' or dem_gradient is None:
            return gm.mean()
        if dem_gradient.shape[-2:] != (H, W):
            dem_gradient = F.interpolate(dem_gradient, size=(H, W),
                                         mode='bilinear', align_corners=False)
        return (gm / (dem_gradient.abs() + self.alpha)).mean()

    def forward_feature_alignment(self, F_chg, F2, dem_gradient=None):
        """Stage 1 -> (F2_aligned, phi, smooth_loss)."""
        if self.alignment_type == 'none':
            H, W = F_chg.shape[-2:]
            F2_aligned = F.interpolate(F2, size=(H, W),
                                       mode='bilinear', align_corners=False)
            phi = torch.zeros(F_chg.shape[0], 2, H, W, device=F_chg.device)
            smooth_loss = torch.tensor(0.0, device=F_chg.device)
            return F2_aligned, phi, smooth_loss
        if self.alignment_type != 'cfda':
            H, W = F_chg.shape[-2:]
            F2_aligned = F.interpolate(F2, size=(H, W),
                                       mode='bilinear', align_corners=False)
            phi = torch.zeros(F_chg.shape[0], 2, H, W, device=F_chg.device)
            return F2_aligned, phi, torch.tensor(0.0, device=F_chg.device)
        phi = self.corr_estimator(F_chg, F2)
        F2_aligned = self._warp(F2, phi)
        smooth_loss = self._smooth_loss(phi, dem_gradient)
        return F2_aligned, phi, smooth_loss

    def compute_vdr_losses(self, M, I2):
        """Stage 2 -> (def_loss, egdr_loss)."""
        def_loss = compute_def_loss(M)
        egdr_loss = compute_egdr_loss(M, I2, mu=F.softplus(self.mu_egdr))
        return def_loss, egdr_loss

    def forward(self, F_chg, F2, dem_gradient=None):
        return self.forward_feature_alignment(F_chg, F2, dem_gradient)
