from pathlib import Path
from collections import Counter
import traceback
import json

import numpy as np
import laspy
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .utils import set_seeds, estimate_ground_z


class NPZDataset(Dataset):
    """Dataset for .npz tiles written by build_dataset_from_las()."""
    def __init__(self, folder):
        folder = Path(folder)
        self.files = sorted(map(str, folder.glob("*.npz")))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {folder}. Did preprocessing run?")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fp = self.files[i]
        try:
            arr = np.load(fp)
            X = (arr["feat"] if "feat" in arr.files else arr["xyz"]).astype(np.float32)
            y = arr["y"].astype(np.int64)
            if X.ndim != 2 or X.shape[0] != y.shape[0]:
                raise ValueError(f"Bad shapes in {fp}: X{X.shape}, y{y.shape}")
            return torch.from_numpy(X), torch.from_numpy(y)
        except Exception as e:
            print(f"\n!! Error reading {fp}: {e}\n")
            traceback.print_exc()
            raise


def build_dataset_from_las(
    *,
    LAS_PATH,
    OUT_DIR,
    SAMPLE_N,
    BLOCK_SIZE,
    STRIDE,
    VAL_SPLIT,
    TEST_SPLIT,
    SEED,
    REPEAT_PER_TILE,
    MIN_PTS_TILE,
    CELL_SIZE,
    QUANTILE,
):
    """Preprocess a labeled LAS into NPZ tiles with features [x0,y0,z0,HAG]."""

    # ---- type coercions (important with reticulate) ----
    LAS_PATH        = str(LAS_PATH)
    OUT_DIR         = Path(OUT_DIR)
    SAMPLE_N        = int(SAMPLE_N)
    BLOCK_SIZE      = float(BLOCK_SIZE)
    STRIDE          = float(STRIDE)
    VAL_SPLIT       = float(VAL_SPLIT)
    TEST_SPLIT      = float(TEST_SPLIT)
    SEED            = int(SEED)
    REPEAT_PER_TILE = int(REPEAT_PER_TILE)
    MIN_PTS_TILE    = int(MIN_PTS_TILE)
    CELL_SIZE       = float(CELL_SIZE)
    QUANTILE        = float(QUANTILE)

    set_seeds(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test"]:
        (OUT_DIR / s).mkdir(exist_ok=True)

    las = laspy.read(LAS_PATH)
    xyz = np.c_[las.x, las.y, las.z].astype(np.float32)

    if hasattr(las, "label"):
        y = np.asarray(las.label, dtype=np.int64); label_field = "label"
    elif hasattr(las, "classification"):
        y = np.asarray(las.classification, dtype=np.int64); label_field = "classification"
    else:
        raise ValueError("No label field found in LAS. Need 'label' or 'classification'.")

    # HAG
    ground_z = estimate_ground_z(xyz, cell=CELL_SIZE, quantile=QUANTILE)
    hag = (xyz[:, 2] - ground_z).astype(np.float32)

    # local frame (subtract global XY min)
    xyz_local = xyz.copy()
    xyz_local[:, :2] -= xyz_local[:, :2].min(0, keepdims=True)

    # sliding grid
    xy = xyz_local[:, :2]
    mins, maxs = xy.min(0), xy.max(0)
    xs = np.arange(mins[0], maxs[0] + 1e-6, STRIDE, dtype=np.float32)
    ys = np.arange(mins[1], maxs[1] + 1e-6, STRIDE, dtype=np.float32)

    tiles = []
    for x0 in xs:
        for y0 in ys:
            m = (xy[:, 0] >= x0) & (xy[:, 0] < x0 + BLOCK_SIZE) & \
                (xy[:, 1] >= y0) & (xy[:, 1] < y0 + BLOCK_SIZE)
            sel = np.where(m)[0]
            if sel.size >= MIN_PTS_TILE:
                tiles.append(sel)
    print("Tiles kept:", len(tiles))
    if not tiles:
        raise RuntimeError("No tiles found (check BLOCK_SIZE/STRIDE/MIN_PTS_TILE).")

    # split tiles
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(tiles))
    tiles = [tiles[i] for i in perm]
    n_val  = int(len(tiles) * VAL_SPLIT)
    n_test = int(len(tiles) * TEST_SPLIT)
    val_tiles   = tiles[:n_val]
    test_tiles  = tiles[n_val:n_val + n_test]
    train_tiles = tiles[n_val + n_test:]
    print("Tile splits:", len(train_tiles), len(val_tiles), len(test_tiles))

    def save_samples(tiles, split):
        out = OUT_DIR / split
        idx_counter = 0
        for sel in tqdm(tiles, desc=f"writing {split}"):
            pts  = xyz_local[sel]
            labs = y[sel].astype(np.int64)
            hsel = hag[sel].astype(np.float32)
            M = len(sel)
            repeat = REPEAT_PER_TILE if M >= SAMPLE_N else 1
            for _ in range(repeat):
                pick  = rng.choice(M, SAMPLE_N, replace=(M < SAMPLE_N))
                p_xyz = pts[pick].astype(np.float32)
                p_hag = hsel[pick][:, None].astype(np.float32)
                # zero-mean center per sample (keep absolute HAG)
                p_xyz -= p_xyz.mean(0, keepdims=True)
                feat = np.hstack([p_xyz, p_hag])   # (SAMPLE_N, 4)
                np.savez(out / f"{idx_counter:08d}.npz", feat=feat, y=labs[pick])
                idx_counter += 1

    save_samples(train_tiles, "train")
    save_samples(val_tiles,   "val")
    save_samples(test_tiles,  "test")

    def split_hist(split):
        hist = Counter()
        for f in (OUT_DIR / split).glob("*.npz"):
            arr = np.load(f)
            hist.update(arr["y"].tolist())
        return {int(k): int(v) for k, v in hist.items()}

    manifest = {
        "label_field": label_field,
        "in_channels": 4,
        "feature_channels": ["x_centered", "y_centered", "z_centered", "hag"],
        "train_hist": split_hist("train"),
        "val_hist":   split_hist("val"),
        "test_hist":  split_hist("test"),
        "sample_n":   SAMPLE_N,
        "block_size": BLOCK_SIZE,
        "stride":     STRIDE,
        "cell_size":  CELL_SIZE,
        "quantile":   QUANTILE,
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Manifest written to:", OUT_DIR / "manifest.json")


def make_loaders(*, OUT_DIR, BATCH_SIZE, debug=True):
    """Build train/val/test dataloaders from OUT_DIR."""
    OUT_DIR   = Path(OUT_DIR)
    BATCH_SIZE = int(BATCH_SIZE)

    train_ds = NPZDataset(OUT_DIR / "train")
    val_ds   = NPZDataset(OUT_DIR / "val")
    test_ds  = NPZDataset(OUT_DIR / "test")

    if debug:
        nw, pin, pw = 0, False, False
    else:
        nw, pin, pw = 2, True, True

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=nw, pin_memory=pin, persistent_workers=pw)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=nw, pin_memory=pin, persistent_workers=pw)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=nw, pin_memory=pin, persistent_workers=pw)

    return train_ds, val_ds, test_ds, train_dl, val_dl, test_dl
