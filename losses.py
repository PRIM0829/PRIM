"""
PRISM Loss Functions

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math as _math
import numpy as np


# ============================================================
# 1. Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2.0,
                 balance_index=0, smooth=1e-5, size_average=True):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        if target.dim() == 4:
            target = torch.squeeze(target, 1)
        target = target.view(-1, 1)

        alpha = self.alpha
        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha_val = alpha
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - alpha_val)
            alpha[self.balance_index] = alpha_val

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        ignore_mask = (idx == 255) | (idx == 225)
        idx[ignore_mask] = 0
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot_key * F.softmax(logit, dim=-1)).sum(1) + self.smooth
        logpt = pt.log()

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), self.gamma) * logpt
        loss = loss[~ignore_mask.squeeze()]

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


# ============================================================
# 2. Tversky Dice Loss (global / patch-wise)
# ============================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=255, alpha=0.7, beta=0.3,
                 local_patch_size=0):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.beta = beta
        self.local_patch_size = local_patch_size

    def forward(self, logits, targets):
        if self.local_patch_size > 0:
            return self._forward_local(logits, targets)
        return self._forward_global(logits, targets)

    def _forward_global(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        probs_fg = probs[:, 1, :, :]
        targets_fg = (targets == 1).float()
        probs_fg = probs_fg.contiguous().view(-1)
        targets_fg = targets_fg.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)
        valid_mask = (targets_flat != self.ignore_index).float()
        probs_fg = probs_fg * valid_mask
        targets_fg = targets_fg * valid_mask
        TP = (probs_fg * targets_fg).sum()
        FP = (probs_fg * (1 - targets_fg)).sum()
        FN = ((1 - probs_fg) * targets_fg).sum()
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - tversky

    def _forward_local(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        probs_fg = probs[:, 1, :, :]
        if targets.dim() == 4:
            targets = targets.squeeze(1)
        targets_fg = (targets == 1).float()
        B, H, W = probs_fg.shape
        ps = self.local_patch_size
        pad_h = (ps - H % ps) % ps
        pad_w = (ps - W % ps) % ps
        if pad_h > 0 or pad_w > 0:
            probs_fg = F.pad(probs_fg, (0, pad_w, 0, pad_h), mode='reflect')
            targets_fg = F.pad(targets_fg, (0, pad_w, 0, pad_h), mode='reflect')
            targets = F.pad(targets.float(), (0, pad_w, 0, pad_h), mode='reflect').long()
            _, H_pad, W_pad = probs_fg.shape
        else:
            H_pad, W_pad = H, W
        num_h, num_w = H_pad // ps, W_pad // ps
        N = num_h * num_w
        probs_patches = probs_fg.reshape(B, num_h, ps, num_w, ps)
        probs_patches = probs_patches.permute(0, 1, 3, 2, 4).reshape(B * N, ps * ps)
        targets_patches = targets_fg.reshape(B, num_h, ps, num_w, ps)
        targets_patches = targets_patches.permute(0, 1, 3, 2, 4).reshape(B * N, ps * ps)
        targets_flat = targets.reshape(B, num_h, ps, num_w, ps)
        targets_flat = targets_flat.permute(0, 1, 3, 2, 4).reshape(B * N, ps * ps)
        valid_mask = (targets_flat != self.ignore_index).float()
        patch_weight = valid_mask.sum(dim=1)
        probs_patches = probs_patches * valid_mask
        targets_patches = targets_patches * valid_mask
        TP = (probs_patches * targets_patches).sum(dim=1)
        FP = (probs_patches * (1 - targets_patches)).sum(dim=1)
        FN = ((1 - probs_patches) * targets_patches).sum(dim=1)
        tversky_per_patch = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth)
        valid_patches = (patch_weight > 0).float()
        if valid_patches.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        loss = (1 - tversky_per_patch) * valid_patches
        return loss.sum() / valid_patches.sum()


# ============================================================
# 3. Change Detection Loss (Focal + Dice + multi-scale)
# ============================================================
class ChangeDetectionLossPhy(nn.Module):
    def __init__(self, lambda_f=1.0, lambda_d=1.0, lambda_ms=0.3, gamma=2.0,
                 focal_alpha=None, dice_alpha=0.5, dice_beta=0.5,
                 dice_local_patch_size=0, ignore_index=255):
        super().__init__()
        if focal_alpha is None:
            focal_alpha = [dice_alpha, dice_beta]
        self.focal = FocalLoss(alpha=focal_alpha, gamma=gamma)
        self.dice = DiceLoss(alpha=dice_alpha, beta=dice_beta,
                             local_patch_size=dice_local_patch_size,
                             ignore_index=ignore_index)
        self.lambda_f = lambda_f
        self.lambda_d = lambda_d
        self.lambda_ms = lambda_ms

    def forward(self, pred, target, aux_preds=None):
        if target.dim() == 4:
            target = target.squeeze(1)
        if pred.shape[-2:] != target.shape[-2:]:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear',
                                 align_corners=True)
        loss_main = self.lambda_f * self.focal(pred, target) + \
                    self.lambda_d * self.dice(pred, target)
        ms_loss = 0.0
        if aux_preds is not None and len(aux_preds) > 0:
            for ap in aux_preds:
                if ap.shape[-2:] != target.shape[-2:]:
                    ap = F.interpolate(ap, size=target.shape[-2:], mode='bilinear',
                                       align_corners=True)
                ms_loss += self.lambda_f * self.focal(ap, target) + \
                           self.lambda_d * self.dice(ap, target)
            ms_loss = ms_loss / len(aux_preds)
        else:
            for s in [0.5, 0.25]:
                h = int(pred.shape[2] * s); w = int(pred.shape[3] * s)
                sp = F.interpolate(pred, size=(h, w), mode='bilinear',
                                   align_corners=True)
                st = F.interpolate(target.unsqueeze(1).float(), size=(h, w),
                                   mode='nearest').squeeze(1).long()
                ms_loss += self.lambda_f * self.focal(sp, st) + \
                           self.lambda_d * self.dice(sp, st)
            ms_loss = ms_loss / 2
        return loss_main + self.lambda_ms * ms_loss


# ============================================================
# 4. Physics-regularization losses
# ============================================================
def _get_gaussian_kernel_2d(kernel_size=5, sigma=1.0, channels=1):
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


class PhysicalConsistencyLossPhy(nn.Module):

    def __init__(self, kernel_size=15, sigma=5.0, T_ref=0.75):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.T_ref = T_ref

    def low_pass_filter(self, x):
        C = x.shape[1]
        kernel = _get_gaussian_kernel_2d(self.kernel_size, self.sigma, C).to(x.device)
        return F.conv2d(x, kernel, padding=self.kernel_size // 2, groups=C)

    def forward(self, F_path_hat, F_A, T_eff_hat):
        with torch.no_grad():
            target_path = self.low_pass_filter(F_A)
        loss_path = F.mse_loss(F_path_hat, target_path)
        if T_eff_hat.dim() == 1:
            T_eff_hat = T_eff_hat.view(1, -1)
        loss_trans = F.mse_loss(T_eff_hat, torch.full_like(T_eff_hat, self.T_ref))
        return loss_path + loss_trans


class OrthogonalDisentanglementLoss(nn.Module):

    def __init__(self, lambda_sp=0.01, lambda_ac=0.1):
        super().__init__()
        self.lambda_sp = lambda_sp
        self.lambda_ac = lambda_ac

    def dag_constraint(self, A):
        d = A.shape[0]
        M = A * A
        exp_M = torch.matrix_exp(M)
        return torch.trace(exp_M) - d

    def forward(self, F_terrain, F_change, A=None):
        B, C, H, W = F_terrain.shape
        N = B * H * W
        F_t = F_terrain.permute(0, 2, 3, 1).contiguous().view(N, C)
        F_c = F_change.permute(0, 2, 3, 1).contiguous().view(N, C)
        cross = torch.mm(F_t.T, F_c)
        orth_loss = (cross ** 2).sum() / (N * C)
        dag_loss = torch.tensor(0.0, device=F_terrain.device)
        if A is not None:
            sparsity = torch.norm(A, p=1)
            acyclic = self.dag_constraint(A)
            dag_loss = self.lambda_sp * sparsity + self.lambda_ac * acyclic
        return {'orth_loss': orth_loss, 'dag_loss': dag_loss,
                'total': orth_loss + dag_loss}


class TerrainSmoothnessLoss(nn.Module):

    def __init__(self, alpha=1e-3):
        super().__init__()
        self.alpha = alpha

    def _gradient(self, x):
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = F.pad(gy, (0, 0, 0, 1))
        gx = F.pad(gx, (0, 1, 0, 0))
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    def forward(self, phi, dem_gradient=None):
        phi_x = phi[:, 0:1, :, :]
        phi_y = phi[:, 1:2, :, :]
        grad_phi_x = self._gradient(phi_x)
        grad_phi_y = self._gradient(phi_y)
        gm = torch.sqrt(grad_phi_x ** 2 + grad_phi_y ** 2 + 1e-8)
        if dem_gradient is not None:
            if dem_gradient.shape[1] > 1:
                dem_gradient = dem_gradient.mean(dim=1, keepdim=True)
            if dem_gradient.shape[2:] != gm.shape[2:]:
                dem_gradient = F.interpolate(dem_gradient, size=gm.shape[2:],
                                             mode='bilinear', align_corners=False)
            weight = 1.0 / (dem_gradient + self.alpha)
        else:
            weight = torch.ones_like(gm)
        return (weight * gm ** 2).mean()


# ============================================================
# 5. Three-phase weight scheduler
# ============================================================
class DynamicWeightScheduler:
    

    def __init__(self, total_epochs, lambda1_range=(1.0, 0.1),
                 lambda2_range=(0.5, 0.05), lambda3_range=(0.1, 0.5),
                 phase_boundaries=(0.3, 0.7), schedule_type='cosine'):
        self.total_epochs = total_epochs
        self.lambda1_range = lambda1_range
        self.lambda2_range = lambda2_range
        self.lambda3_range = lambda3_range
        self.p1, self.p2 = phase_boundaries
        self.schedule_type = schedule_type

    def _interpolate(self, ratio, start, end):
        if self.schedule_type == 'cosine':
            w = 0.5 * (1.0 + _math.cos(_math.pi * ratio))
            return end + (start - end) * w
        return start + (end - start) * ratio

    def get_weights(self, epoch):
        if self.schedule_type == 'static':
            return {'lambda_phys': self.lambda1_range[0],
                    'lambda_orth': self.lambda2_range[0],
                    'lambda_smooth': self.lambda3_range[0]}
        progress = epoch / max(self.total_epochs - 1, 1)
        if progress <= self.p1:
            l1, l2, l3 = self.lambda1_range[0], self.lambda2_range[0], self.lambda3_range[0]
        elif progress <= self.p2:
            ratio = (progress - self.p1) / (self.p2 - self.p1)
            l1 = self._interpolate(ratio, *self.lambda1_range)
            l2 = self._interpolate(ratio, *self.lambda2_range)
            l3 = self._interpolate(ratio, *self.lambda3_range)
        else:
            l1, l2, l3 = self.lambda1_range[1], self.lambda2_range[1], self.lambda3_range[1]
        return {'lambda_phys': l1, 'lambda_orth': l2, 'lambda_smooth': l3}


# ============================================================
# 6. Total loss
# ============================================================
class PhyCDNetTotalLoss(nn.Module):


    def __init__(self, total_epochs=100, focal_alpha=None, dice_alpha=0.5,
                 dice_beta=0.5, dice_local_patch_size=0, T_ref=0.75,
                 ignore_index=255, lambda_f=1.0, lambda_d=1.0, lambda_ms=0.3,
                 ablation=None, loss_schedule='three_phase', orth_alpha=0.1,
                 lambda_phys_range=(3.0, 0.3), lambda_cov=1.0,
                 lambda_orth_range=(0.01, 0.1), lambda_smooth_range=(0.1, 0.5),
                 lambda_entropy=0.5, lambda_geo=0.5):
        super().__init__()
        self.cd_loss = ChangeDetectionLossPhy(
            lambda_f=lambda_f, lambda_d=lambda_d, lambda_ms=lambda_ms,
            focal_alpha=focal_alpha, dice_alpha=dice_alpha, dice_beta=dice_beta,
            dice_local_patch_size=dice_local_patch_size,
            ignore_index=ignore_index)
        self.phys_loss = PhysicalConsistencyLossPhy(T_ref=T_ref)
        self.orth_loss = OrthogonalDisentanglementLoss()
        self.smooth_loss = TerrainSmoothnessLoss()
        self.lambda_cov = lambda_cov
        self.lambda_entropy = lambda_entropy
        self.lambda_geo = lambda_geo
        if ablation == 'no_orth' or orth_alpha == 0:
            orth_range = (0.0, 0.0)
        else:
            orth_range = (lambda_orth_range[0], orth_alpha)
        sched_type = 'static' if loss_schedule == 'static' else 'cosine'
        self.scheduler = DynamicWeightScheduler(
            total_epochs=total_epochs,
            lambda1_range=lambda_phys_range,
            lambda2_range=orth_range,
            lambda3_range=lambda_smooth_range,
            schedule_type=sched_type)
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, pred, target, aux_preds=None, F_path_hat=None, F_A=None,
                T_eff_hat=None, F_terrain=None, F_change=None, A=None,
                phi=None, dem_gradient=None,
                def_loss=None, egdr_loss=None, R_cov=None, R_phys=None):
        weights = self.scheduler.get_weights(self.current_epoch)
        loss_cd = self.cd_loss(pred, target, aux_preds)
        lp = self.phys_loss(F_path_hat, F_A, T_eff_hat) \
            if (F_path_hat is not None and F_A is not None and T_eff_hat is not None) \
            else torch.tensor(0.0, device=pred.device)
        lo = torch.tensor(0.0, device=pred.device)
        ld = torch.tensor(0.0, device=pred.device)
        if F_terrain is not None and F_change is not None:
            od = self.orth_loss(F_terrain, F_change, A)
            lo = od['orth_loss']
            ld = od['dag_loss']
        ls = self.smooth_loss(phi, dem_gradient) if phi is not None \
            else torch.tensor(0.0, device=pred.device)
        le = def_loss if def_loss is not None else torch.tensor(0.0, device=pred.device)
        lg = egdr_loss if egdr_loss is not None else torch.tensor(0.0, device=pred.device)
        lcov = R_cov if R_cov is not None else torch.tensor(0.0, device=pred.device)
        lrphys = R_phys if R_phys is not None else torch.tensor(0.0, device=pred.device)
        total = (loss_cd
                 + weights.get('lambda_phys', 1.0) * (lp + lrphys)
                 + self.lambda_cov * lcov
                 + weights.get('lambda_orth', 0.1) * lo
                 + weights.get('lambda_smooth', 0.1) * ls
                 + self.lambda_entropy * le
                 + self.lambda_geo * lg)
        return {'total': total, 'cd_loss': loss_cd, 'phys_loss': lp,
                'orth_loss': lo, 'dag_loss': ld, 'smooth_loss': ls,
                'def_loss': le, 'egdr_loss': lg,
                'R_cov': lcov, 'R_phys': lrphys, 'weights': weights}
