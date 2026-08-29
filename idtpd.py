

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ITPDC(nn.Module):
    

    def __init__(self, r=7, sigma=3.0, descriptors='full'):
        super().__init__()
        self.r = r
        self.sigma = sigma
        self.descriptors = descriptors
        # Shadow probability head (Eq. 29)
        self.shadow_cnn = nn.Sequential(
            nn.Conv2d(5, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1), nn.Sigmoid())

    def forward(self, I2):
        
        B, C, H, W = I2.shape
        I2_gray = I2.mean(dim=1, keepdim=True)
        S = self.shadow_cnn(torch.cat([I2, I2_gray, I2_gray], dim=1))
        # Placeholder channels (full derivation in manuscript)
        A = torch.zeros_like(S)
        cosT = torch.ones_like(S)
        sinT = torch.zeros_like(S)
        Hc = torch.zeros_like(S)
        D = torch.cat([S, A, cosT, sinT, Hc], dim=1)
        if self.descriptors != 'full':
            if self.descriptors == 'S':
                D[:, 1:] = 0.0
            elif self.descriptors == 'S,A':
                D[:, 2:] = 0.0
            elif self.descriptors == 'S,A,Theta':
                D[:, 4:] = 0.0
        return D


class TAGC(nn.Module):
    

    def __init__(self, n_superpixels=500, k_nn=8, sigma_D=None,
                 graph_type='knn_spatial', spatial_radius=3):
        super().__init__()
        self.n_superpixels = n_superpixels
        self.k_nn = k_nn
        self.sigma_D = sigma_D
        self.graph_type = graph_type
        self.spatial_radius = spatial_radius

    def forward(self, D, I2):
  
        B, _, H, W = D.shape
        gh = int(np.sqrt(self.n_superpixels * H / W))
        gw = self.n_superpixels // gh
        cell_h, cell_w = H // gh, W // gw
        yi = (torch.arange(H, device=D.device).view(-1, 1).repeat(1, W) // cell_h)
        xi = (torch.arange(W, device=D.device).view(1, -1).repeat(H, 1) // cell_w)
        sp_id = (yi.clamp(0, gh - 1) * gw + xi.clamp(0, gw - 1)).long()
        sp_id = sp_id.unsqueeze(0).expand(B, -1, -1)
        N = gh * gw

        W_list = []
        for b in range(B):
            D_b = D[b]
            sp_b = sp_id[b]
            nodes = torch.zeros(N, 5, device=D.device)
            for n in range(N):
                mask = (sp_b == n)
                if mask.sum() > 0:
                    nodes[n] = D_b[:, mask].mean(dim=1)
            dist = ((nodes.unsqueeze(0) - nodes.unsqueeze(1)) ** 2).sum(dim=2)
            sigma = self.sigma_D or max(dist.median(), 0.1)
            W_list.append(torch.exp(-dist / (sigma * sigma + 1e-6)))
        W = torch.stack(W_list, dim=0)
        return W, sp_id, N


class DRGCD(nn.Module):


    def __init__(self, in_dim, hidden_dim=128, out_dim=256):
        super().__init__()
        self.gcn1 = nn.Linear(in_dim, hidden_dim)
        self.gcn2 = nn.Linear(hidden_dim, out_dim * 2)
        self.out_dim = out_dim

    def forward(self, Z, W):
        B, N, _ = Z.shape
        W_tilde = W + torch.eye(N, device=W.device).unsqueeze(0)
        deg = W_tilde.sum(dim=2).clamp(min=1)
        deg_inv_sqrt = deg.pow(-0.5).diag_embed()
        A_norm = deg_inv_sqrt @ W_tilde @ deg_inv_sqrt
        H1 = F.relu(A_norm @ self.gcn1(Z))
        H2 = A_norm @ self.gcn2(H1)
        F_ter = H2[:, :, :self.out_dim]
        F_chg = H2[:, :, self.out_dim:]
        return F_ter, F_chg


class IDTPDModule(nn.Module):


    def __init__(self, channels=256, n_superpixels=100, k_nn=8,
                 gate_type='ours', terrain_dim=3, d=32,
                 descriptors='full', graph_type='knn_spatial'):
        super().__init__()
        self.channels = channels
        self.gate_type = gate_type
        if gate_type == 'none':
            self.itpdc = None
            self.tagc = None
            self.drgcd = None
        else:
            self.itpdc = ITPDC(r=7, sigma=3.0, descriptors=descriptors)
            self.tagc = TAGC(n_superpixels=n_superpixels, k_nn=k_nn,
                             graph_type=graph_type)
            self.drgcd = DRGCD(in_dim=channels * 2, hidden_dim=128,
                               out_dim=channels)

    def _superpixel_pool(self, feat, sp_id, N):
        B, C, H, W = feat.shape
        sp_flat = sp_id.view(B, -1).long()
        onehot = F.one_hot(sp_flat, num_classes=N).float().permute(0, 2, 1)
        feat_flat = feat.view(B, C, -1)
        Z = torch.bmm(feat_flat, onehot.transpose(1, 2))
        area = onehot.sum(dim=2).clamp(min=1).unsqueeze(1)
        return (Z / area).transpose(1, 2)

    def _superpixel_unpool(self, node_feat, sp_id, H, W):
        B, N, C = node_feat.shape
        sp_flat = sp_id.view(B, -1).long()
        onehot = F.one_hot(sp_flat, num_classes=N).float()
        out_flat = torch.bmm(node_feat.transpose(1, 2), onehot.permute(0, 2, 1))
        return out_flat.view(B, C, H, W)

    def forward(self, F1_corr, I2=None):
        """F1_corr: [B,C,H,W] -> (F_chg, F_ter, orth_loss, dag_loss)."""
        B, C, H, W = F1_corr.shape
        dag_loss = torch.tensor(0.0, device=F1_corr.device)
        if self.gate_type == 'none':
            return F1_corr, torch.zeros_like(F1_corr), \
                   torch.tensor(0.0, device=F1_corr.device), dag_loss
        if I2 is None:
            I2_use = F1_corr[:, :3] if C >= 3 else F1_corr[:, :1].repeat(1, 3, 1, 1)
        else:
            I2_use = I2
        D = self.itpdc(I2_use)
        W_sp, sp_id_img, N_img = self.tagc(D, I2_use)
        Hf, Wf = F1_corr.shape[-2:]
        sp_id_feat = F.interpolate(sp_id_img.float().unsqueeze(1),
                                   size=(Hf, Wf), mode='nearest').squeeze(1).long()
        active_ids = torch.unique(sp_id_feat.reshape(-1))
        N = len(active_ids)
        id_map = {int(old): new for new, old in enumerate(active_ids.cpu().tolist())}
        sp_id_remap = torch.zeros_like(sp_id_feat)
        for old_id, new_id in id_map.items():
            sp_id_remap[sp_id_feat == old_id] = new_id
        active_idx = active_ids.to(W_sp.device)
        W_sp = W_sp[:, active_idx][:, :, active_idx]
        Z_nodes = self._superpixel_pool(F1_corr, sp_id_remap, N)
        Z_nodes_2x = torch.cat([Z_nodes, Z_nodes], dim=2)
        F_ter_nodes, F_chg_nodes = self.drgcd(Z_nodes_2x, W_sp)
        F_ter = self._superpixel_unpool(F_ter_nodes, sp_id_remap, Hf, Wf)
        F_chg = self._superpixel_unpool(F_chg_nodes, sp_id_remap, Hf, Wf)
        F_t_flat = F_ter.view(B, C, -1)
        F_c_flat = F_chg.view(B, C, -1)
        cross_corr = torch.bmm(F_t_flat, F_c_flat.transpose(1, 2))
        orth_loss = (cross_corr ** 2).mean()
        return F_chg, F_ter, orth_loss, dag_loss
