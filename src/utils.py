# inst/python/vegseg/utils.py
import random, numpy as np, torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); 
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def estimate_ground_z(xyz: np.ndarray, cell: float = 0.25, quantile: float = 0.05):
    """Grid the XY plane with cell size (m). Per cell, take z-quantile as ground."""
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    x0, y0 = x.min(), y.min()
    ix = np.floor((x - x0)/cell).astype(np.int32)
    iy = np.floor((y - y0)/cell).astype(np.int32)
    ny = iy.max() + 1
    gid = ix * ny + iy

    order = np.argsort(gid)
    gid_s, z_s = gid[order], z[order]
    u, start = np.unique(gid_s, return_index=True)
    end = np.r_[start[1:], len(gid_s)]

    ground = np.full(ix.max()*ny + ny, np.nan, np.float32)
    for k, s in enumerate(start):
        e = end[k]
        ground[u[k]] = np.quantile(z_s[s:e], quantile)

    g = ground[gid]
    if np.isnan(g).any():
        good = ~np.isnan(g)
        nbrs = NearestNeighbors(n_neighbors=1).fit(xyz[good,:2])
        _, j = nbrs.kneighbors(xyz[~good,:2])
        g[~good] = g[good][j[:,0]]
    return g.astype(np.float32)
