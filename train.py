import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ImageNet pretrained weights from timm. Both reach about 80% ImageNet top-1 with a
# similar parameter count (22M vs 24M), so the comparison is about the architecture,
# not about one backbone having stronger pretraining.
BACKBONES = {
    "deit": "deit_small_patch16_224.fb_in1k",
    "resnet": "resnet50.a1_in1k",
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- data

def read_labels(csv_path: Path):
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [(r["subject"], r["path"], float(r["pitch"]), float(r["yaw"])) for r in reader]


class GazeDataset(Dataset):
    def __init__(self, root: Path, rows, train: bool):
        self.root = root
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        sid, path, pitch, yaw = self.rows[i]
        img = cv2.imread(str(self.root / path))                  # BGR uint8, 224x224
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.train:
            # Horizontal flip mirrors the face, so the gaze mirrors too: yaw -> -yaw.
            if torch.rand(1).item() < 0.5:
                img = img[:, ::-1]
                yaw = -yaw
            # Mild brightness / contrast jitter (webcam lighting varies a lot).
            a = 0.8 + 0.4 * torch.rand(1).item()
            b = -0.1 + 0.2 * torch.rand(1).item()
            img = np.clip(img * a + b, 0.0, 1.0)

        img = (img - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
        y = torch.tensor([pitch, yaw], dtype=torch.float32)
        return x, y, sid


# ---------------------------------------------------------------- metric

def pitchyaw_to_vector(py: torch.Tensor) -> torch.Tensor:
    """(N, 2) pitch/yaw in radians -> (N, 3) unit gaze vectors (inverse of preprocess.py)."""
    p, y = py[:, 0], py[:, 1]
    return torch.stack([-torch.cos(p) * torch.sin(y),
                        -torch.sin(p),
                        -torch.cos(p) * torch.cos(y)], dim=1)


def angular_error_deg(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    cos = (pitchyaw_to_vector(pred) * pitchyaw_to_vector(true)).sum(dim=1)
    return torch.rad2deg(torch.acos(cos.clamp(-1.0, 1.0)))


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    per_subject = defaultdict(list)
    for x, y, sids in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = model(x)
        err = angular_error_deg(pred.float().cpu(), y)
        for s, e in zip(sids, err.tolist()):
            per_subject[s].append(e)

    subject_means = {s: float(np.mean(v)) for s, v in sorted(per_subject.items())}
    return {
        "mean_over_subjects": float(np.mean(list(subject_means.values()))),
        "mean_over_images": float(np.mean([e for v in per_subject.values() for e in v])),
        "per_subject": subject_means,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=BACKBONES, required=True)
    ap.add_argument("--test", nargs="+", required=True, help="held-out participants, e.g. p00 p01")
    ap.add_argument("--data", default="data/normalized")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.5, help="warmup length in epochs")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--subset", type=int, default=None,
                    help="use only N train and N test images (quick smoke test)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None   # RTX 30xx supports bf16

    # ---- split by participant (never mix one person's images across train and test)
    root = Path(args.data)
    rows = read_labels(root / "labels.csv")
    test_set = set(args.test)
    train_rows = [r for r in rows if r[0] not in test_set]
    test_rows = [r for r in rows if r[0] in test_set]
    if args.subset:
        rng = np.random.default_rng(args.seed)
        train_rows = [train_rows[i] for i in rng.permutation(len(train_rows))[:args.subset]]
        test_rows = [test_rows[i] for i in rng.permutation(len(test_rows))[:args.subset]]
    print(f"train: {len(train_rows)} images | test: {len(test_rows)} images ({' '.join(sorted(test_set))})")

    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.workers > 0)
    train_loader = DataLoader(GazeDataset(root, train_rows, train=True), batch_size=args.bs,
                              shuffle=True, drop_last=True, **loader_kw)
    test_loader = DataLoader(GazeDataset(root, test_rows, train=False), batch_size=args.bs * 2,
                             shuffle=False, **loader_kw)

    # ---- model: pretrained backbone, classification head replaced by a 2-output regression head
    model = timm.create_model(BACKBONES[args.backbone], pretrained=not args.no_pretrained, num_classes=2)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model: {BACKBONES[args.backbone]} ({n_params:.1f}M params) on {device}")

    # ---- optimizer: AdamW, linear warmup then cosine decay, stepped every batch
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(args.warmup * len(train_loader)))

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    loss_fn = nn.L1Loss()

    run_dir = Path(args.out) / f"{args.backbone}_test-{'-'.join(sorted(test_set))}"
    run_dir.mkdir(parents=True, exist_ok=True)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(x)
            loss = loss_fn(pred.float(), y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_time = time.time() - t0
        result = evaluate(model, test_loader, device, amp_dtype)
        history.append({"epoch": epoch, "train_l1": running / seen, **result})
        print(f"epoch {epoch:2d} | train L1 {running / seen:.4f} rad | "
              f"test error {result['mean_over_subjects']:.2f} deg | "
              f"{seen / train_time:.0f} img/s, {train_time / 60:.1f} min")

    # Report the LAST epoch, not the best one: picking the best epoch by test error
    # would be tuning on the test set.
    final = history[-1]
    print("per-subject test error (deg):",
          ", ".join(f"{s} {e:.2f}" for s, e in final["per_subject"].items()))
    print(f"FINAL mean angular error: {final['mean_over_subjects']:.2f} deg")

    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"args": vars(args), "history": history}, f, indent=2)
    print(f"saved to {run_dir}")


if __name__ == "__main__":
    main()
