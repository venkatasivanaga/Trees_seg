# inst/python/vegseg/train.py
from pathlib import Path
import torch
from torch import nn
from tqdm import tqdm

from .model import HeightAwarePointNetTiny
from .dataset import make_loaders

__all__ = ["train_model"]

def train_model(config: dict):
    """
    Train the HeightAwarePointNetTiny model using parameters from `config`.
    Returns a dict: {"best_oa": float, "best_epoch": int, "ckpt_path": str}
    """
    # ---- pull config (with safe casts) ----
    out_dir      = Path(config["out_dir"])
    model_path   = Path(config.get("model_path", out_dir / "model" / "best_model.pth"))
    batch_size   = int(config.get("batch_size", 16))
    epochs       = int(config.get("epochs", 20))
    lr           = float(config.get("learning_rate", 1e-5))
    weight_decay = float(config.get("weight_decay", 1e-4))
    num_classes  = int(config.get("num_classes", 3))
    device       = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # ---- data loaders ----
    train_ds, val_ds, _, train_dl, val_dl, _ = make_loaders(
        OUT_DIR=out_dir, BATCH_SIZE=batch_size, debug=True
    )

    # ---- model/loss/opt ----
    X0, _ = train_ds[0]
    in_ch = X0.shape[-1]
    model = HeightAwarePointNetTiny(
        in_ch=in_ch, num_classes=num_classes, k=16,
        z_idx=2, hag_idx=(3 if in_ch >= 4 else None), use_height_prior=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_oa, best_epoch = -1.0, -1
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- train loop ----
    for epoch in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        for X, y in tqdm(train_dl, desc=f"train {epoch}/{epochs}", leave=False):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits.reshape(-1, num_classes), y.reshape(-1))
            optim.zero_grad(); loss.backward(); optim.step()
            run_loss += loss.item()

        # validation OA
        model.eval(); correct = tot = 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                pred = model(X).argmax(-1)
                correct += (pred == y).sum().item()
                tot     += y.numel()
        val_oa = correct / max(tot, 1)

        print(f"Epoch {epoch:02d} | lr {lr:g} | train loss {run_loss/len(train_dl):.4f} | val OA {val_oa:.4f}")

        if val_oa > best_oa:
            best_oa, best_epoch = val_oa, epoch
            torch.save(model.state_dict(), str(model_path))

    return {"best_oa": best_oa, "best_epoch": best_epoch, "ckpt_path": str(model_path)}
