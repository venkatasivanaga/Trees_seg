from .config_bridge import apply_cfg
from . import infer, model as mdl
import torch

def run_predict(py_cfg, new_in_las, new_out_las, mode="overwrite"):
    class C: pass
    cfg = apply_cfg(py_cfg, C)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model:", cfg.model_path)
    m = mdl.HeightAwarePointNetTiny(in_ch=4, num_classes=3, k=16,
                                    z_idx=2, hag_idx=3, use_height_prior=True).to(device)
    state = torch.load(cfg.model_path, map_location=device)
    m.load_state_dict(state); m.eval()

    print(f"Inference on {new_in_las}...")
    y_pred = infer.infer_on_las_path(
        m, new_in_las,
        BLOCK_SIZE=cfg.block_size, STRIDE=cfg.stride,
        SAMPLE_N=cfg.sample_n, REPEAT_PER_TILE=cfg.repeat_per_tile,
        MIN_PTS_TILE=cfg.min_pts_tile,
        CELL_SIZE=cfg.cell_size, QUANTILE=cfg.quantile,
        DEVICE=device
    )
    infer.write_predictions_to_las(new_in_las, new_out_las, y_pred, mode=mode)
    print("Wrote predictions to:", new_out_las)
    return new_out_las
