# Model definitions for HeightAwarePointNetTiny

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn_idx(coords, k: int):
    """ Compute the indices of the k nearest neighbors for each point in coords. """
    with torch.no_grad():
        B, N, _ = coords.shape
        xx = (coords ** 2).sum(-1, keepdim=True)
        dist = xx + xx.transpose(1, 2) - 2 * coords @ coords.transpose(1, 2)
        dist = torch.clamp(dist, min=0)
        _, idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
    return idx

def gather_batched(x, idx):
    """ Gather the neighbors' features based on indices. """
    B, N, C = x.shape  # B: batch size, N: number of points, C: feature channels
    k = idx.shape[-1]  # Number of neighbors
    idx_flat = (idx + (torch.arange(B, device=idx.device).view(B, 1, 1) * N)).reshape(-1)
    x_flat = x.reshape(B * N, C)
    out = x_flat[idx_flat].reshape(B, N, k, C)
    return out

class HeightMixer(nn.Module):
    def __init__(self, in_ch: int, z_idx=2, hag_idx=3):
        super().__init__()
        self.z_idx = z_idx
        self.hag_idx = hag_idx if hag_idx is not None and hag_idx < in_ch else None
        self.a = nn.Parameter(torch.tensor(1.0))
        self.c = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(1.0)) if self.hag_idx is not None else None
        
    def forward(self, x):
        z = x[..., self.z_idx]
        if self.hag_idx is None:
            return self.a * z + self.c
        hag = x[..., self.hag_idx]
        return self.a * z + self.b * hag + self.c


class HeightPriorBias(nn.Module):
    def __init__(self, hag_idx=3, init_thresh=0.15, init_sharp=10.0, init_scale=0.5):
        super().__init__()
        self.hag_idx = hag_idx
        self.thresh  = nn.Parameter(torch.tensor(init_thresh))
        self.sharp   = nn.Parameter(torch.tensor(init_sharp))
        self.scale   = nn.Parameter(torch.tensor(init_scale))
        
    def forward(self, x, logits, red_class=0):
        if self.hag_idx is None or self.hag_idx >= x.shape[-1]:
            return logits
        hag = x[..., self.hag_idx]
        bias = self.scale * torch.sigmoid(self.sharp * (self.thresh - hag))
        logits[..., red_class] = logits[..., red_class] + bias
        return logits


class LocalAggBlock(nn.Module):
    def __init__(self, in_feat_ch, out_feat_ch, k=16, pool="max"):
        super().__init__()
        self.k = k
        self.pool = pool
        self.mlp = nn.Sequential(
            nn.Linear(in_feat_ch + in_feat_ch + 3, out_feat_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, coords_knn, feat):
        idx = knn_idx(coords_knn, self.k)
        nb_f = gather_batched(feat, idx)
        nb_p = gather_batched(coords_knn, idx)
        fi = feat.unsqueeze(2).expand_as(nb_f)
        pi = coords_knn.unsqueeze(2).expand_as(nb_p)
        x = torch.cat([fi, nb_f - fi, nb_p - pi], dim=-1)
        B, N, k, D = x.shape
        x = x.reshape(B * N * k, D)
        x = self.mlp(x)
        x = x.reshape(B, N, k, -1)
        x = torch.max(x, dim=2)[0] if self.pool == "max" else torch.mean(x, dim=2)
        return x


class HeightAwarePointNetTiny(nn.Module):
    def __init__(self, in_ch=4, num_classes=3, k=16, widths=(64, 128, 256),
                 z_idx=2, hag_idx=3, use_height_prior=True):
        super().__init__()
        self.hmix = HeightMixer(in_ch, z_idx=z_idx, hag_idx=hag_idx)
        self.hprior = HeightPriorBias(hag_idx=hag_idx) if use_height_prior else None
        self.use_height_prior = use_height_prior

        self.stem = nn.Sequential(nn.Linear(in_ch, widths[0]), nn.ReLU(inplace=True))
        self.block1 = LocalAggBlock(widths[0], widths[1], k=k, pool="max")
        self.block2 = LocalAggBlock(widths[1], widths[2], k=k, pool="max")
        self.glob = nn.Sequential(nn.Linear(widths[2], widths[2]), nn.ReLU(inplace=True))
        self.head = nn.Sequential(
            nn.Linear(widths[2]*2, widths[2]), nn.ReLU(inplace=True),
            nn.Linear(widths[2], num_classes)
        )

    def forward(self, x):  # x: (B, N, in_ch)
        z_eff = self.hmix(x)  # (B, N)
        coords_knn = torch.stack([x[..., 0], x[..., 1], z_eff], -1)  # (B, N, 3)
        f = self.stem(x)
        f = self.block1(coords_knn, f)
        f = self.block2(coords_knn, f)
        g = torch.max(f, dim=1)[0]
        g = self.glob(g).unsqueeze(1).expand(-1, x.size(1), -1)
        logits = self.head(torch.cat([f, g], dim=-1))
        if self.use_height_prior and (self.hprior is not None):
            logits = self.hprior(x, logits, red_class=0)
        return logits
