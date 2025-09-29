#!/usr/bin/env python3
"""
train.py — Train selected image classification models and save .pth weights.

Supported models:
  - resnet50               (torchvision)
  - shufflenet_v2_x1_0     (torchvision)
  - regnety_8gf            (timm: regnety_008.pycls_in1k)
  - swin_t                 (timm: swin_tiny_patch4_window7_224.ms_in22k)

Saves:
  ./models/<model>.pth           # state_dict only
  ./models/<model>_meta.json     # {"classes": [...], "img_size": 224, "model": "<name>"}

Dataset layout:
  data_root/
    train/<class>/*.jpg
    val/<class>/*.jpg
    test/<class>/*.jpg
"""

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Optional timm backbones for RegNetY-8GF and Swin-T
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


# ------------------------- Model factory  -------------------------
def build_model(name: str, num_classes: int) -> nn.Module:
    """
    Build one of: resnet50, shufflenet_v2_x1_0 (torchvision) or regnety_8gf, swin_t (timm).
    """
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
        raise ValueError(f"{name} requires 'timm' (pip install timm).")

    timm_ids = {
        "regnety_8gf": "regnety_008.pycls_in1k",
        "swin_t": "swin_tiny_patch4_window7_224.ms_in22k",
    }
    if n not in timm_ids:
        raise ValueError(f"Unsupported model '{name}'. Choose from: resnet50, shufflenet_v2_x1_0, regnety_8gf, swin_t")
    return timm.create_model(timm_ids[n], pretrained=False, num_classes=num_classes)


# ------------------------- Data pipeline -------------------------
def make_loaders(data_dir: str, img_size: int, batch_size: int, workers: int = 4):
    """
    Build train/val/test dataloaders. Uses light augmentations on train; center-crop on val/test.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(Path(data_dir) / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(Path(data_dir) / "val",   transform=eval_tf)
    test_ds  = datasets.ImageFolder(Path(data_dir) / "test",  transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes


# ------------------------- Eval helper -------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader, device, criterion):
    """
    Evaluate on a loader. Returns (loss, acc, precision_macro, recall_macro, f1_macro).
    """
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    preds_all, labels_all = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x)
            loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(1)
        correct += (preds == y).sum().item()
        total += y.numel()
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    avg_loss = loss_sum / max(total, 1)
    acc = correct / max(total, 1)
    prec = precision_score(labels_all, preds_all, average="macro", zero_division=0)
    rec  = recall_score(labels_all, preds_all, average="macro", zero_division=0)
    f1   = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return avg_loss, acc, prec, rec, f1


# ------------------------- Training loop -------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, classes = make_loaders(
        args.data_dir, args.img_size, args.batch_size, args.workers
    )
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    model = build_model(args.model, num_classes).to(device)

    # Standard recipe: CrossEntropy + AdamW + cosine decay on LR plateau (ReduceLROnPlateau)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_loss)

        print(f"[{epoch+1}/{args.epochs}] "
              f"TrainLoss={train_loss:.4f} | "
              f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} ValF1={val_f1:.4f}")

        # Early stopping on accuracy improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    # Load best weights and test once
    model.load_state_dict(best_weights)
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate(model, test_loader, device, criterion)
    print(f"\nFinal Test — Loss: {test_loss:.4f}  Acc: {test_acc:.4f}  F1: {test_f1:.4f}")

    # Save .pth + meta
    out_dir = Path(args.models_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pth_path = out_dir / f"{args.model}.pth"
    torch.save(model.state_dict(), pth_path)
    print(f"Saved weights: {pth_path}")

    meta = {"classes": classes, "img_size": args.img_size, "model": args.model}
    meta_path = out_dir / f"{args.model}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved meta:    {meta_path}")


# ------------------------- CLI -------------------------
def main():
    ap = argparse.ArgumentParser(description="Train ResNet-50 / RegNetY-8GF / ShuffleNetV2-x1.0 / Swin-T and save .pth")
    ap.add_argument("--data_dir", type=str, required=True, help="Dataset root with train/ val/ test/")
    ap.add_argument("--models_dir", type=str, default="./models", help="Output dir for .pth files")
    ap.add_argument("--model", type=str, required=True,
                    choices=["resnet50", "regnety_8gf", "shufflenet_v2_x1_0", "swin_t"],
                    help="Choose the model to train")
    ap.add_argument("--img_size", type=int, default=224, help="Input size (e.g., 224)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=6, help="Early stopping patience (epochs without val acc improvement)")
    args = ap.parse_args()

    train(args)


if __name__ == "__main__":
    main()
