"""
PRISM — Physics-Guided Causal Change Detection via A2N Unrolling
==================================================================
Three core modules: PCRA (Sec. 3.2), IDTPD (Sec. 3.3), VDR (Sec. 3.4).

Pipeline:
  I_A --[Degradation Simulator]--> F1 --[PCRA]--> F1_corr --[IDTPD]--> F_chg
                                                                        │
  I_B --------------------------> F2 ---------------[VDR align]----> F2_aligned
                                                                        │
                         P = |F_chg - F2_aligned| --[Decoder]--> M (DEF+EGDR)

Usage:
  from models import PRISM
  model = PRISM(feature_dim=256, n_class=2)
  output = model(image_A, image_B)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .pcra import PCRAModule
from .idtpd import IDTPDModule
from .vdr import VDRModule


# ============================================================
# Degradation Simulator (cross-sensor preconditioning)
# ============================================================
class DegradationSimulator(nn.Module):

    def __init__(self, max_translation=15, max_rotation=5.0,
                 downsample_factor=4.0, img_size=256, seed=None):
        super().__init__()
        self.max_translation = max_translation
        self.max_rotation = max_rotation
        self.downsample_factor = downsample_factor
        self.img_size = img_size
        self.seed = seed
        self._counter = 0

    def _random_affine(self, B, H, W, device):
        if self.seed is not None:
            g = torch.Generator(device=device)
            affines = []
            for b in range(B):
                g.manual_seed(self.seed + self._counter * 1000 + b)
                angle_deg = (torch.rand(1, generator=g, device=device).item() * 2 - 1) * self.max_rotation
                theta = angle_deg * (math.pi / 180.0)
                cos_a = math.cos(theta); sin_a = math.sin(theta)
                tx = (torch.rand(1, generator=g, device=device).item() * 2 - 1) * self.max_translation / (W / 2.0)
                ty = (torch.rand(1, generator=g, device=device).item() * 2 - 1) * self.max_translation / (H / 2.0)
                aff = torch.tensor([[cos_a, -sin_a, tx], [sin_a, cos_a, ty]], device=device)
                affines.append(aff)
            self._counter += 1
            return torch.stack(affines, dim=0)
        angle_deg = (torch.rand(B, device=device) * 2 - 1) * self.max_rotation
        theta = angle_deg * (math.pi / 180.0)
        cos_a = torch.cos(theta); sin_a = torch.sin(theta)
        tx = (torch.rand(B, device=device) * 2 - 1) * self.max_translation / (W / 2.0)
        ty = (torch.rand(B, device=device) * 2 - 1) * self.max_translation / (H / 2.0)
        affine = torch.zeros(B, 2, 3, device=device)
        affine[:, 0, 0] = cos_a;  affine[:, 0, 1] = -sin_a; affine[:, 0, 2] = tx
        affine[:, 1, 0] = sin_a;  affine[:, 1, 1] =  cos_a; affine[:, 1, 2] = ty
        return affine

    def forward(self, x, label=None):
        B, C, H, W = x.shape
        affine = self._random_affine(B, H, W, x.device)
        grid = F.affine_grid(affine, x.size(), align_corners=False)
        x_warped = F.grid_sample(x, grid, mode='bilinear', align_corners=False)
        x_down = F.interpolate(x_warped, scale_factor=1.0 / self.downsample_factor,
                               mode='bilinear', align_corners=False)
        x_simulated = F.interpolate(x_down, size=(H, W), mode='bilinear',
                                    align_corners=False)
        label_out = None
        if label is not None:
            label_out = F.grid_sample(label.float(), grid, mode='nearest',
                                      align_corners=False)
        return x_simulated, affine, label_out


# ============================================================
# Encoder / Decoder
# ============================================================
def get_gaussian_kernel_phy(kernel_size=5, sigma=1.0, channels=1):
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


class LowPassFilterPhy(nn.Module):

    def __init__(self, channels, kernel_size=11, sigma=3.0):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        kernel = get_gaussian_kernel_phy(kernel_size, sigma, channels)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.kernel_size // 2,
                        groups=self.channels)


class DualResNetEncoder(nn.Module):
   
    def __init__(self, pretrained=True, output_channels=256):
        super().__init__()
        import torchvision.models as tv_models
        resnet_a = tv_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.conv1_a = resnet_a.conv1; self.bn1_a = resnet_a.bn1
        self.relu_a = resnet_a.relu; self.maxpool_a = resnet_a.maxpool
        self.layer1_a = resnet_a.layer1; self.layer2_a = resnet_a.layer2
        self.layer3_a = resnet_a.layer3; self.layer4_a = resnet_a.layer4

        resnet_b = tv_models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.conv1_b = resnet_b.conv1; self.bn1_b = resnet_b.bn1
        self.relu_b = resnet_b.relu; self.maxpool_b = resnet_b.maxpool
        self.layer1_b = resnet_b.layer1; self.layer2_b = resnet_b.layer2
        self.layer3_b = resnet_b.layer3; self.layer4_b = resnet_b.layer4

        self.compress_a = nn.Conv2d(512, output_channels, kernel_size=1)
        self.compress_b = nn.Conv2d(512, output_channels, kernel_size=1)
        self.output_channels = output_channels

    def forward(self, I_A, I_B):
        x = self.conv1_a(I_A); x = self.bn1_a(x); x = self.relu_a(x); x = self.maxpool_a(x)
        x = self.layer1_a(x); x = self.layer2_a(x); x = self.layer3_a(x)
        F1 = self.compress_a(self.layer4_a(x))
        y = self.conv1_b(I_B); y = self.bn1_b(y); y = self.relu_b(y); y = self.maxpool_b(y)
        y = self.layer1_b(y); y = self.layer2_b(y); y = self.layer3_b(y)
        F2 = self.compress_b(self.layer4_b(y))
        return F1, F2


class DifferenceDecoder(nn.Module):
    """Decoder: |F_chg - F2_aligned| -> change logits."""
    def __init__(self, in_channels=256, mid_channels=128, out_channels=2):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels // 2), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 2, out_channels, kernel_size=1))

    def forward(self, F_chg, F2_aligned, target_size=None):
        out = self.decoder(torch.abs(F_chg - F2_aligned))
        if target_size is not None:
            out = F.interpolate(out, size=target_size, mode='bilinear',
                                align_corners=False)
        return out


# ============================================================
# PRISM
# ============================================================
class PRISM(nn.Module):



    def __init__(self, backbone='resnet18', feature_dim=256, n_class=2,
                 pretrained=True, use_dem=False, ablation=None,
                 # --- PCRA ---
                 crma_correction='pdfn', crma_dilate_rate=3,
                 pcra_correction=None, pcra_dilate_rate=None,
                 pcra_confidence='spce', pcra_tau=0.1,
                 pcra_smoother='fixed', pcra_fixed_sigma=3.0,
                 # --- IDTPD ---
                 pcod_gate_type='ours', idtpd_gate_type=None,
                 idtpd_descriptors='full', idtpd_graph_type='knn_spatial',
                 idtpd_n_superpixels=100, idtpd_k_nn=8,
                 # --- VDR ---
                 cfda_alignment='cfda', cfda_pyramid_levels=3,
                 cfda_temperature=0.07, cfda_smoothness='terrain',
                 cfda_smoothness_kernel=17,
                 vdr_alignment=None, vdr_pyramid_levels=None,
                 vdr_temperature=None, vdr_smoothness=None,
                 vdr_smoothness_kernel=None,
                 vdr_mu=1.0, vdr_mu_learnable=True,
                 eval_seed=None):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_dem = use_dem

        # CLI resolution: new names take priority over legacy names
        _pcra_corr = pcra_correction if pcra_correction is not None else crma_correction
        _pcra_rate = pcra_dilate_rate if pcra_dilate_rate is not None else crma_dilate_rate
        _idtpd_gate = idtpd_gate_type if idtpd_gate_type is not None else pcod_gate_type
        _vdr_align = vdr_alignment if vdr_alignment is not None else cfda_alignment
        _vdr_L = vdr_pyramid_levels if vdr_pyramid_levels is not None else cfda_pyramid_levels
        _vdr_tau = vdr_temperature if vdr_temperature is not None else cfda_temperature
        _vdr_sm = vdr_smoothness if vdr_smoothness is not None else cfda_smoothness
        _vdr_smk = vdr_smoothness_kernel if vdr_smoothness_kernel is not None else cfda_smoothness_kernel

        # Ablation set
        _abl = (ablation or '').strip()
        if _abl == 'baseline':
            _abl_set = {'no_crma', 'no_pcod', 'no_cfda',
                        'no_pcra', 'no_idtpd', 'no_vdr'}
        else:
            _abl_set = set(a.strip() for a in _abl.split(',') if a.strip())
        self.ablation = _abl
        self._abl_set = _abl_set

        self.encoder = DualResNetEncoder(pretrained=pretrained,
                                         output_channels=feature_dim)

        # ---- PCRA ----
        eff_pcra_corr = _pcra_corr if ('no_crma' not in _abl_set and
                                       'no_pcra' not in _abl_set) else 'none'
        eff_pcra_rate = _pcra_rate if ('no_crma' not in _abl_set and
                                       'no_pcra' not in _abl_set) else 0
        self.pcra = PCRAModule(channels=feature_dim, correction_type=eff_pcra_corr,
                               dilate_rate=eff_pcra_rate,
                               confidence_type=pcra_confidence, hard_tau=pcra_tau,
                               smoother_type=pcra_smoother, fixed_sigma=pcra_fixed_sigma)

        # ---- IDTPD ----
        eff_idtpd_gate = _idtpd_gate if ('no_pcod' not in _abl_set and
                                         'no_idtpd' not in _abl_set) else 'none'
        self.idtpd = IDTPDModule(channels=feature_dim, terrain_dim=3, d=32,
                                 gate_type=eff_idtpd_gate,
                                 descriptors=idtpd_descriptors,
                                 graph_type=idtpd_graph_type,
                                 n_superpixels=idtpd_n_superpixels,
                                 k_nn=idtpd_k_nn)

        # ---- VDR ----
        eff_vdr_align = _vdr_align if ('no_cfda' not in _abl_set and
                                       'no_vdr' not in _abl_set) else 'none'
        self.vdr = VDRModule(channels=feature_dim, alignment_type=eff_vdr_align,
                             pyramid_levels=_vdr_L, temperature=_vdr_tau,
                             smoothness_type=_vdr_sm, smoothness_kernel=_vdr_smk,
                             mu_egdr=vdr_mu, mu_learnable=vdr_mu_learnable)

        self.decoder = DifferenceDecoder(in_channels=feature_dim,
                                         mid_channels=128, out_channels=n_class)
        self.low_pass = LowPassFilterPhy(channels=feature_dim,
                                         kernel_size=11, sigma=3.0)
        self.simulator = DegradationSimulator(max_translation=15, max_rotation=5.0,
                                              downsample_factor=4.0, img_size=256,
                                              seed=eval_seed)

    def forward(self, I_A, I_B, dem_data=None, label=None):
        I1_original = I_A.clone()
        terrain_map = None
        dem_gradient = None
        if dem_data is not None:
            if isinstance(dem_data, dict):
                terrain_map = dem_data.get('terrain', None)
                dem_gradient = dem_data.get('gradient', None)
            elif isinstance(dem_data, torch.Tensor):
                terrain_map = dem_data

        # Degradation simulation on the pre-event image
        I_A, affine_mat, label_warped = self.simulator(I_A, label=label)
        I1_simulated = I_A.clone()
        F1, F2 = self.encoder(I_A, I_B)

        # ---- P1: PCRA (Physics-Constrained Radiometric Alignment) ----
        if 'no_crma' in self._abl_set or 'no_pcra' in self._abl_set:
            F1_corr = F1
            F_path = torch.zeros_like(F1)
            T_eff = torch.ones(1, self.feature_dim, 1, 1, device=F1.device)
            R_cov = torch.tensor(0.0, device=F1.device)
            R_phys = torch.tensor(0.0, device=F1.device)
        else:
            F1_corr, F_path, T_eff, R_cov, R_phys, _w = self.pcra(F1, F2)

        # ---- P2: IDTPD (Image-Derived Terrain Proxy Decoupling) ----
        if 'no_pcod' in self._abl_set or 'no_idtpd' in self._abl_set:
            F_chg = F1_corr
            F_ter = torch.zeros_like(F1_corr)
            orth_loss = torch.tensor(0.0, device=F1.device)
            dag_loss = torch.tensor(0.0, device=F1.device)
        else:
            F_chg, F_ter, orth_loss, dag_loss = self.idtpd(F1_corr, I_B)

        # ---- P3: VDR Stage 1 (feature-flow alignment) ----
        if 'no_cfda' in self._abl_set or 'no_vdr' in self._abl_set:
            F2_aligned = F2
            phi = torch.zeros(F2.shape[0], 2, F2.shape[2], F2.shape[3],
                              device=F2.device)
            smooth_loss = torch.tensor(0.0, device=F2.device)
        else:
            F2_aligned, phi, smooth_loss = self.vdr(F_chg, F2, dem_gradient)

        # Decoder: |F_chg - F2_aligned| -> change logits P
        P = self.decoder(F_chg, F2_aligned,
                         target_size=(I_B.shape[-2], I_B.shape[-1]))

        # ---- VDR Stage 2: DEF + EGDR on probability map M ----
        M = torch.sigmoid(P[:, 1:2, :, :])
        if 'no_cfda' in self._abl_set or 'no_vdr' in self._abl_set:
            def_loss = torch.tensor(0.0, device=F1.device)
            egdr_loss = torch.tensor(0.0, device=F1.device)
        else:
            def_loss, egdr_loss = self.vdr.compute_vdr_losses(M, I_B)

        # Physical consistency loss
        lf_F1 = self.low_pass(F1)
        phys_loss = F.mse_loss(F_path, lf_F1.detach()) + \
                    F.mse_loss(torch.abs(T_eff), torch.ones_like(T_eff) * 0.8) + \
                    R_phys + R_cov

        return {
            'pred': P,
            'M': M,
            'phys_loss': phys_loss, 'orth_loss': orth_loss,
            'dag_loss': dag_loss, 'smooth_loss': smooth_loss,
            'def_loss': def_loss, 'egdr_loss': egdr_loss,
            'R_cov': R_cov, 'R_phys': R_phys,
            'F_path_hat': F_path,
            'F1': F1, 'F2': F2,
            'F2_aligned': F2_aligned, 'T_eff_hat': T_eff,
            'F_ter': F_ter, 'F_chg': F_chg,
            'phi': phi, 'dem_gradient': dem_gradient, 'A': None,
            'label': label_warped,
            'I1_original': I1_original, 'I1_simulated': I1_simulated,
            'F1_corr': F1_corr.detach(),
            'affine_mat': affine_mat,
            # legacy aliases
            'F_A': F1, 'F_B': F2,
            'F_B_aligned': F2_aligned,
            'F_terrain': F_ter, 'F_change': F_chg,
            'F_A_rad': F1_corr.detach(),
            'I_A_original': I1_original, 'I_A_simulated': I1_simulated,
        }
