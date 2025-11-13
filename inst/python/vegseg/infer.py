# inst/python/vegseg/infer.py
from pathlib import Path  # <-- REQUIRED
import numpy as np
import laspy
import torch
import sklearn.neighbors as skn

from .utils import estimate_ground_z

__all__ = ["infer_on_las_path", "write_predictions_to_las"]

@torch.no_grad()
def infer_on_las_path(
    model,
    las_path: str,
    *,
    BLOCK_SIZE,
    STRIDE,
    SAMPLE_N,
    REPEAT_PER_TILE,
    MIN_PTS_TILE,
    CELL_SIZE,
    QUANTILE,
    DEVICE=None
):
    """Return per-point predictions for `las_path` using the trained model."""
    DEVICE = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")

    las_path = str(las_path)
    BLOCK_SIZE      = float(BLOCK_SIZE)
    STRIDE          = float(STRIDE)
    SAMPLE_N        = int(SAMPLE_N)
    REPEAT_PER_TILE = int(REPEAT_PER_TILE)
    MIN_PTS_TILE    = int(MIN_PTS_TILE)
    CELL_SIZE       = float(CELL_SIZE)
    QUANTILE        = float(QUANTILE)

    print(f"Reading LAS from {las_path} ...")
    las = laspy.read(las_path)
    xyz = np.c_[las.x, las.y, las.z].astype(np.float32)
    N   = xyz.shape[0]

    print("Recomputing HAG ...")
    ground_z = estimate_ground_z(xyz, cell=CELL_SIZE, quantile=QUANTILE)
    hag = (xyz[:, 2] - ground_z).astype(np.float32)

    # local XY like training
    xyz_local = xyz.copy()
    xyz_local[:, :2] -= xyz_local[:, :2].min(0, keepdims=True)

    xy = xyz_local[:, :2]
    mins, maxs = xy.min(0), xy.max(0)
    xs = np.arange(mins[0], maxs[0] + 1e-6, STRIDE, dtype=np.float32)
    ys = np.arange(mins[1], maxs[1] + 1e-6, STRIDE, dtype=np.float32)

    # infer channels from model (assumes .stem[0] is Linear)
    in_ch = model.stem[0].in_features if hasattr(model, "stem") else 4

    votes = np.zeros((N, int(getattr(model, "head")[-1].out_features)), dtype=np.int32)
    model.eval()

    for x0 in xs:
        for y0 in ys:
            sel = np.where((xy[:, 0] >= x0) & (xy[:, 0] < x0 + BLOCK_SIZE) &
                           (xy[:, 1] >= y0) & (xy[:, 1] < y0 + BLOCK_SIZE))[0]
            M = sel.size
            if M < MIN_PTS_TILE:
                continue
            reps = REPEAT_PER_TILE if M >= SAMPLE_N else 1
            for _ in range(reps):
                pick = np.random.choice(M, SAMPLE_N, replace=(M < SAMPLE_N))
                idx  = sel[pick]
                pts  = xyz_local[idx].astype(np.float32)
                pts -= pts.mean(0, keepdims=True)
                feat = np.hstack([pts, hag[idx][:, None]])  # (SAMPLE_N, 4)

                t = torch.from_numpy(feat[None, ...]).to(DEVICE)
                pred = model(t).argmax(-1).squeeze(0).cpu().numpy().astype(np.int16)
                np.add.at(votes, (idx, pred), 1)

    has_vote = votes.sum(1) > 0
    y_pred = np.full(N, -1, dtype=np.int16)
    y_pred[has_vote] = votes[has_vote].argmax(1).astype(np.int16)

    # fill any unvoted with 1-NN in XY
    left = np.where(~has_vote)[0]
    if left.size:
        right = np.where(has_vote)[0]
        nn = skn.NearestNeighbors(n_neighbors=1).fit(xyz[right, :2])
        _, j = nn.kneighbors(xyz[left, :2])
        y_pred[left] = y_pred[right[j[:, 0]]]

    print("Prediction complete.")
    return y_pred


def write_predictions_to_las(in_las: str, out_las: str, y_pred: np.ndarray, *, mode="overwrite"):
    """Write predictions to LAS. mode='overwrite' or 'extra' (adds pred_label)."""
    from laspy import ExtraBytesParams

    in_las  = str(in_las)
    out_las = str(out_las)

    las = laspy.read(in_las)
    assert y_pred.shape[0] == len(las.x), "Predictions length must match number of points."

    if mode == "overwrite":
        las.classification = np.clip(y_pred, 0, 255).astype(np.uint8)
    elif mode == "extra":
        if "pred_label" not in las.point_format.dimension_names:
            las.add_extra_dim(ExtraBytesParams(name="pred_label", type=np.uint16))
        las.pred_label = y_pred.astype(np.uint16)
    else:
        raise ValueError("mode must be 'overwrite' or 'extra'")

    Path(out_las).parent.mkdir(parents=True, exist_ok=True)
    las.write(out_las)
    print(f"Wrote predictions to: {out_las}")
