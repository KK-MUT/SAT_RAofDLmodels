#!/usr/bin/env python3
"""
eval_models.py — Evaluate all trained models (.pth) on the test subset.

• Expects each model saved by train.py:
      ./models/resnet50.pth
      ./models/resnet50_meta.json   # contains {"classes": [...], "img_size": 224, "model": "resnet50"}

• Computes:
      - Accuracy
      - Precision_macro
      - Recall_macro
      - F1_macro
      - Confusion matrix
• Saves:
      ./eval_results/<model>_report.txt
      ./eval_results/<model>_confusion_matrix.csv

Example
-------
python eval_models.py --data_dir ./dataset --models_dir ./models
"""

import argparse
import json
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Optional timm for RegNetY-8GF and Swin-T
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


# ------------------------- Model factory (same 4 models) -------------------------
def build_model(name: str, num_classes: int) -> nn.Module:
    n = name.lower()
    if n == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if n == "shufflenet_v2_x1_0":
        m = models.shufflenet_v2_x1_0(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if not TIMM_AVAILABLE:
        raise ValueError(f"{name} requires timm (pip install timm)")
    timm_map = {
        "regnety_8gf": "regnety_008.pycls_in1k",
        "swin_t": "swin_tiny_patch4_window7_224.ms_in22k",
    }
    return timm.create_model(timm_map[name], pretrained=False, num_classes=num_classes)


# ------------------------- Dataloader -------------------------
def make_test_loader(data_dir: str, img_size: int, batch: int, workers: int = 4):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = datasets.ImageFolder(Path(data_dir) / "test", transform=tf)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return loader, ds.classes


# ------------------------- Evaluation -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, device, classes):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    acc  = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm   = confusion_matrix(all_labels, all_preds, labels=list(range(len(classes))))
    return acc, prec, rec, f1, cm, all_labels, all_preds


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Evaluate all .pth models on test subset")
    ap.add_argument("--data_dir", required=True, help="Dataset root with test/ subfolder")
    ap.add_argument("--models_dir", default="./models", help="Folder containing .pth and _meta.json")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="./eval_results", help="Folder for saving reports")
    args = ap.parse_args()

    device = torch.device(args.device)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect all pth files
    model_files = sorted(Path(args.models_dir).glob("*.pth"))
    if not model_files:
        raise FileNotFoundError(f"No .pth files in {args.models_dir}")

    for pth in model_files:
        name = pth.stem
        meta_path = pth.with_name(f"{name}_meta.json")
        if not meta_path.exists():
            print(f"[SKIP] {name}: missing meta JSON → skipped.")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        classes = meta["classes"]
        img_size = meta.get("img_size", 224)
        arch = meta.get("model", name)

        print(f"\n=== Evaluating {name} ({arch}) ===")
        test_loader, ds_classes = make_test_loader(args.data_dir, img_size, args.batch_size, args.workers)

        if ds_classes != classes:
            print("[WARN] class order in dataset differs from model meta; using meta order for metrics.")

        model = build_model(arch, num_classes=len(classes)).to(device)
        sd = torch.load(pth, map_location=device)
        try:
            model.load_state_dict(sd, strict=True)
        except Exception as e:
            print(f"[WARN] strict load failed: {e} → retrying with strict=False")
            model.load_state_dict(sd, strict=False)

        acc, prec, rec, f1, cm, y_true, y_pred = evaluate(model, test_loader, device, classes)
        print(f"Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}")

        # Save text report
        rep_path = out_root / f"{name}_report.txt"
        with rep_path.open("w", encoding="utf-8") as f:
            f.write(f"Model: {name}\nClasses: {classes}\n\n")
            f.write(f"Accuracy:  {acc:.4f}\n")
            f.write(f"Precision: {prec:.4f}\n")
            f.write(f"Recall:    {rec:.4f}\n")
            f.write(f"F1_macro:  {f1:.4f}\n\n")
            f.write("Confusion matrix (rows=true, cols=pred):\n")
            np.savetxt(f, cm, fmt="%d")

        # Save confusion matrix separately as CSV
        cm_path = out_root / f"{name}_confusion_matrix.csv"
        np.savetxt(cm_path, cm, fmt="%d", delimiter=",")
        print(f"Saved → {rep_path}\n         {cm_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
