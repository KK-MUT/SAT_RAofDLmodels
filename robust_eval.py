#!/usr/bin/env python3
"""
robust_eval.py — Adversarial robustness evaluation for models saved as .pth (state_dict).

Features
- FGSM and PGD (L∞) white-box attacks (per-model) + optional transfer evaluation
- Robust accuracy RA(ε), normalized AUC, ε50 and ε90 thresholds
- Saves per-model curves (CSV), summary (JSON), and plots (PNG)
- Handles torchvision/timm models; loads meta JSON saved at training (classes, img_size, model name)

Usage examples
-------------
# Evaluate all .pth in ./models with default epsilons (in 1/255 units), per-model attacks
python robust_eval.py --data_dir ./dataset --models_dir ./models --select "*.pth"

# Custom epsilon grid (1/255 units) and PGD only, 10 steps, step=ε/4
python robust_eval.py --data_dir ./dataset --models_dir ./models --attacks pgd \
  --eps_from255 0,0.25,0.5,1.0,2.0 --pgd_steps 10 --pgd_alpha_scale 0.25

# Transfer: craft on the first matched model and evaluate all models on those adversarials
python robust_eval.py --data_dir ./dataset --models_dir ./models --transfer_source_index 0

# Missing meta JSON? Provide fallback arch/classes/img_size
python robust_eval.py --data_dir ./dataset --models_dir ./models --select "resnet50.pth" \
  --model_fallback resnet50 --classes_csv "A10,F15,F16,MiG21" --img_size 224
"""

import argparse
import glob
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Optional timm (ConvNeXt / RegNet / ViT / Swin)
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


# ------------------------- Model builders -------------------------
def build_model(model_name: str, num_classes: int) -> nn.Module:
    name = model_name.lower()
    # torchvision
    if name == "resnet50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if name == "densenet121":
        m = models.densenet121(weights=None); m.classifier = nn.Linear(m.classifier.in_features, num_classes); return m
    if name == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None); m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes); return m
    if name == "shufflenet_v2_x1_0":
        m = models.shufflenet_v2_x1_0(weights=None); m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    if name == "efficientnet_b1":
        m = models.efficientnet_b1(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes); return m
    if name == "vgg16":
        m = models.vgg16(weights=None); m.classifier[6] = nn.Linear(m.classifier[6].in_features, num_classes); return m
    # timm
    if not TIMM_AVAILABLE:
        raise ValueError(f"{model_name} requires timm (pip install timm).")
    timm_map = {
        "regnety_8gf": "regnety_008.pycls_in1k",
        "convnext_tiny": "convnext_tiny.fb_in22k",
        "vit_b": "vit_base_patch16_224.augreg_in21k",
        "swin_t": "swin_tiny_patch4_window7_224.ms_in22k",
    }
    backbone = timm_map.get(name, name)
    return timm.create_model(backbone, pretrained=False, num_classes=num_classes)


# ------------------------- Data loading (returns RAW [0,1]) -------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

def build_transforms_raw(img_size: int):
    # No normalization here; we keep raw [0,1] for correct ε in pixel space
    return transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # [0,1]
    ])

def normalize_in_model_space(x_raw: torch.Tensor) -> torch.Tensor:
    # x_raw in [0,1]; normalize with ImageNet stats
    return (x_raw - IMAGENET_MEAN.to(x_raw.device)) / IMAGENET_STD.to(x_raw.device)

def get_loaders_raw(data_dir: str, img_size: int, batch: int, workers: int):
    tf = build_transforms_raw(img_size)
    test_ds = datasets.ImageFolder(Path(data_dir) / "test", transform=tf)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False, num_workers=workers, pin_memory=True)
    return test_loader, test_ds.classes


# ------------------------- Attacks (operate in RAW space [0,1]) -------------------------
def clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)

def fgsm(x_raw: torch.Tensor, y: torch.Tensor, model: nn.Module, eps: float) -> torch.Tensor:
    """
    FGSM L∞ in raw pixel space: x_adv = clamp(x + eps * sign(grad_x))
    Gradient computed through the normalization + model.
    """
    x = x_raw.clone().detach().requires_grad_(True)
    logits = model(normalize_in_model_space(x))
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    x_adv = clamp01(x + eps * x.grad.data.sign())
    return x_adv.detach()

def pgd_linf(x_raw: torch.Tensor, y: torch.Tensor, model: nn.Module,
             eps: float, steps: int, alpha: float, random_start: bool = True) -> torch.Tensor:
    """
    PGD L∞ in raw pixel space with projection to ε-ball around the original x_raw.
    """
    x0 = x_raw.detach()
    if random_start:
        x = x0 + torch.empty_like(x0).uniform_(-eps, eps)
        x = clamp01(x)
    else:
        x = x0.clone()

    for _ in range(steps):
        x.requires_grad_(True)
        logits = model(normalize_in_model_space(x))
        loss = nn.CrossEntropyLoss()(logits, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            x = x + alpha * x.grad.sign()
            # project to L∞ ball
            x = torch.max(torch.min(x, x0 + eps), x0 - eps)
            x = clamp01(x)
    return x.detach()


# ------------------------- Metrics helpers -------------------------
def interp_eps_at_ra(target_ra: float, eps: np.ndarray, ra: np.ndarray) -> float:
    """
    Smallest epsilon where RA falls below target_ra, using linear interpolation between points.
    Returns np.nan if RA never drops below target within the scanned range.
    """
    # Ensure eps ascending:
    order = np.argsort(eps)
    eps = eps[order]; ra = ra[order]
    if np.all(ra >= target_ra):
        return np.nan
    # find first i with ra[i] < target
    idx = np.argmax(ra < target_ra)
    if idx == 0:
        return eps[0]
    # linear interpolate between (idx-1) and idx
    x0, y0 = eps[idx-1], ra[idx-1]
    x1, y1 = eps[idx], ra[idx]
    if y0 == y1:
        return x1
    t = (target_ra - y0) / (y1 - y0)
    return float(x0 + t * (x1 - x0))

def normalized_auc(eps: np.ndarray, ra: np.ndarray) -> float:
    """
    AUC under RA(eps) normalized by the epsilon range so it's in [0,1] if RA in [0,1].
    """
    order = np.argsort(eps)
    eps = eps[order]; ra = ra[order]
    area = np.trapz(ra, eps)
    denom = (eps[-1] - eps[0]) if len(eps) > 1 else 1.0
    return float(area / denom)


# ------------------------- Core evaluation -------------------------
@torch.no_grad()
def eval_clean_accuracy(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for x_raw, y in loader:
        x_raw, y = x_raw.to(device), y.to(device)
        logits = model(normalize_in_model_space(x_raw))
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)

def eval_robust_curve(model: nn.Module, loader, device: torch.device,
                      attack: str, epsilons: List[float],
                      pgd_steps: int, pgd_alpha_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (eps_array, ra_array) for given model and attack over test split.
    """
    eps_arr = np.array(epsilons, dtype=np.float32)
    ra_list = []
    for eps in eps_arr:
        correct = total = 0
        for x_raw, y in loader:
            x_raw = x_raw.to(device); y = y.to(device)
            if eps == 0.0:
                x_adv = x_raw
            else:
                if attack == "fgsm":
                    x_adv = fgsm(x_raw, y, model, eps)
                elif attack == "pgd":
                    alpha = pgd_alpha_scale * eps
                    x_adv = pgd_linf(x_raw, y, model, eps, steps=pgd_steps, alpha=alpha, random_start=True)
                else:
                    raise ValueError(f"Unknown attack: {attack}")
            logits = model(normalize_in_model_space(x_adv))
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        ra = correct / max(total, 1)
        ra_list.append(ra)
        print(f"  {attack.upper()} eps={eps:.6f}  RA={ra:.4f}")
    return eps_arr, np.array(ra_list, dtype=np.float32)

def eval_transfer_curve(source_model: nn.Module, target_model: nn.Module, loader, device,
                        attack: str, epsilons: List[float],
                        pgd_steps: int, pgd_alpha_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Craft adversarials on source_model, evaluate on target_model.
    """
    eps_arr = np.array(epsilons, dtype=np.float32)
    ra_list = []
    for eps in eps_arr:
        correct = total = 0
        for x_raw, y in loader:
            x_raw = x_raw.to(device); y = y.to(device)
            if eps == 0.0:
                x_adv = x_raw
            else:
                if attack == "fgsm":
                    x_adv = fgsm(x_raw, y, source_model, eps)
                elif attack == "pgd":
                    alpha = pgd_alpha_scale * eps
                    x_adv = pgd_linf(x_raw, y, source_model, eps, steps=pgd_steps, alpha=alpha, random_start=True)
                else:
                    raise ValueError(f"Unknown attack: {attack}")
            logits = target_model(normalize_in_model_space(x_adv))
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
        ra = correct / max(total, 1)
        ra_list.append(ra)
        print(f"  TRANSFER {attack.upper()} eps={eps:.6f}  RA={ra:.4f}")
    return eps_arr, np.array(ra_list, dtype=np.float32)


# ------------------------- IO helpers -------------------------
def save_curve_csv(out_dir: Path, model_name: str, attack: str, eps: np.ndarray, ra: np.ndarray):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{model_name}_{attack}_curve.csv"
    with p.open("w", encoding="utf-8") as f:
        f.write("epsilon,robust_accuracy\n")
        for e, r in zip(eps, ra):
            f.write(f"{e:.8f},{r:.6f}\n")
    return p

def save_plot_png(out_dir: Path, title: str, curves: List[Tuple[str, np.ndarray, np.ndarray]]):
    """
    curves: list of (label, eps, ra)
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5))
    for label, eps, ra in curves:
        plt.plot(eps, ra, marker='o', linewidth=1.5, label=label)
    plt.xlabel("epsilon (in raw pixel scale [0..1])")
    plt.ylabel("Robust Accuracy")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    out = out_dir / (title.lower().replace(" ", "_") + ".png")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out

def save_summary_json(out_dir: Path, fname: str, payload: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / fname
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Adversarial robustness evaluation for .pth models")
    ap.add_argument("--data_dir", type=str, required=True, help="Dataset root with test/ subfolder")
    ap.add_argument("--models_dir", type=str, required=True, help="Folder with .pth weights and optional *_meta.json")
    ap.add_argument("--select", type=str, default="*.pth", help="Glob for model files, e.g. '*.pth' or 'resnet50.pth'")
    ap.add_argument("--out_dir", type=str, default="./robust_outputs", help="Output folder for CSV/PNG/JSON")

    # Fallbacks if meta is missing
    ap.add_argument("--model_fallback", type=str, default=None, help="Architecture for raw .pth (e.g., resnet50, regnety_8gf, swin_t)")
    ap.add_argument("--classes_csv", type=str, default=None, help="Comma-separated class names if meta missing")
    ap.add_argument("--img_size", type=int, default=224, help="Fallback input size if meta missing")

    # Attacks & epsilon grid
    ap.add_argument("--attacks", type=str, default="fgsm,pgd", help="Comma-separated: fgsm,pgd")
    ap.add_argument("--eps_from255", type=str, default="0,0.25,0.5,1.0,2.0", help="Comma-separated eps in 1/255 units")
    ap.add_argument("--pgd_steps", type=int, default=10, help="PGD iterations")
    ap.add_argument("--pgd_alpha_scale", type=float, default=0.25, help="PGD step size alpha = scale * eps")

    # Batch/device
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Transfer evaluation
    ap.add_argument("--transfer_source_index", type=int, default=None,
                    help="If set, craft with this model index (from sorted file list) and evaluate transfer on all models")

    args = ap.parse_args()
    device = torch.device(args.device)
    out_root = Path(args.out_dir); out_root.mkdir(parents=True, exist_ok=True)

    # Epsilon list: convert 1/255 → raw pixel [0..1]
    eps_255 = [float(x) for x in args.eps_from255.split(",") if x.strip() != ""]
    eps_list = [e/255.0 for e in eps_255]
    if 0.0 not in eps_list:
        eps_list = [0.0] + eps_list

    # Collect model files
    paths = sorted(glob.glob(str(Path(args.models_dir) / args.select)))
    if not paths:
        raise FileNotFoundError(f"No models matched '{args.select}' in {args.models_dir}")
    print("Models:")
    for i, p in enumerate(paths):
        print(f"  [{i}] {p}")

    # Classes & img_size resolution per model (meta if available)
    models = []
    all_classes: Optional[List[str]] = None
    resolved_img_size: Optional[int] = None

    for p in paths:
        pth = Path(p)
        meta_path = pth.with_name(pth.stem + "_meta.json")
        model_name = None; classes = None; img_size = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_name = meta.get("model", None)
            classes = meta.get("classes", None)
            img_size = meta.get("img_size", None)
        # fallbacks
        if classes is None and args.classes_csv:
            classes = [c.strip() for c in args.classes_csv.split(",") if c.strip() != ""]
        if img_size is None:
            img_size = args.img_size
        if model_name is None and args.model_fallback:
            model_name = args.model_fallback
        if model_name is None:
            raise ValueError(f"Cannot determine architecture for {p}. Provide --model_fallback or meta JSON.")

        num_classes = len(classes) if classes is not None else None
        if num_classes is None:
            raise ValueError(f"No class list for {p}. Provide classes in meta or via --classes_csv.")

        # Load model
        model = build_model(model_name, num_classes=num_classes).to(device)
        sd = torch.load(p, map_location=device)
        try:
            model.load_state_dict(sd, strict=True)
        except Exception as e:
            print(f"[WARN] strict load failed for {p}: {e}; trying strict=False")
            model.load_state_dict(sd, strict=False)
        model.eval()

        models.append({"path": p, "name": Path(p).stem, "model": model,
                       "classes": classes, "img_size": img_size, "arch": model_name})

        # Enforce consistent classes across models
        if all_classes is None: all_classes = classes
        else:
            if classes != all_classes:
                raise ValueError(f"Class order mismatch: {p} differs from previous models.")

        if resolved_img_size is None: resolved_img_size = img_size
        else: resolved_img_size = max(resolved_img_size, img_size)

    assert all_classes is not None and resolved_img_size is not None
    num_classes = len(all_classes)

    # Data loader (RAW [0,1])
    test_loader, ds_classes = get_loaders_raw(args.data_dir, resolved_img_size, args.batch_size, args.workers)
    if ds_classes != all_classes:
        print("[WARN] Class names in dataset/test differ from model meta. Proceeding with model meta order for metrics.")

    # Clean accuracy
    for m in models:
        acc_clean = eval_clean_accuracy(m["model"], test_loader, device)
        print(f"[CLEAN] {m['name']}  acc={acc_clean:.4f}")

    # Attacks to run
    attacks = [a.strip().lower() for a in args.attacks.split(",") if a.strip()]

    # Per-model robustness (white-box)
    per_model_results = {}
    for m in models:
        per_attack = {}
        for atk in attacks:
            print(f"\n=== {m['name']} | {atk.upper()} ===")
            eps_arr, ra_arr = eval_robust_curve(
                m["model"], test_loader, device, atk, eps_list,
                pgd_steps=args.pgd_steps, pgd_alpha_scale=args.pgd_alpha_scale
            )
            auc = normalized_auc(eps_arr, ra_arr)
            eps50 = interp_eps_at_ra(0.50, eps_arr, ra_arr)
            eps90 = interp_eps_at_ra(0.90, eps_arr, ra_arr)
            per_attack[atk] = {
                "eps": eps_arr.tolist(),
                "ra": ra_arr.tolist(),
                "auc_norm": auc,
                "eps50": eps50,
                "eps90": eps90
            }
            # Save CSV + plot
            out_dir = Path(args.out_dir) / "per_model" / m["name"]
            csv_path = save_curve_csv(out_dir, m["name"], atk, eps_arr, ra_arr)
            png_path = save_plot_png(out_dir, f"{m['name']} {atk.upper()} RA", [(atk.upper(), eps_arr, ra_arr)])
            print(f"Saved: {csv_path} | {png_path}")
        per_model_results[m["name"]] = {"arch": m["arch"], "attacks": per_attack}

    # Transfer evaluation (optional)
    transfer_results = {}
    if args.transfer_source_index is not None:
        src_idx = int(args.transfer_source_index)
        if not (0 <= src_idx < len(models)):
            raise IndexError("--transfer_source_index out of range.")
        src = models[src_idx]
        for i, tgt in enumerate(models):
            label = f"{Path(src['path']).stem}→{Path(tgt['path']).stem}"
            per_attack = {}
            for atk in attacks:
                print(f"\n=== TRANSFER {label} | {atk.upper()} ===")
                eps_arr, ra_arr = eval_transfer_curve(
                    src["model"], tgt["model"], test_loader, device, atk, eps_list,
                    pgd_steps=args.pgd_steps, pgd_alpha_scale=args.pgd_alpha_scale
                )
                auc = normalized_auc(eps_arr, ra_arr)
                eps50 = interp_eps_at_ra(0.50, eps_arr, ra_arr)
                eps90 = interp_eps_at_ra(0.90, eps_arr, ra_arr)
                per_attack[atk] = {
                    "eps": eps_arr.tolist(),
                    "ra": ra_arr.tolist(),
                    "auc_norm": auc,
                    "eps50": eps50,
                    "eps90": eps90
                }
                out_dir = Path(args.out_dir) / "transfer" / f"{Path(src['path']).stem}_to_{Path(tgt['path']).stem}"
                csv_path = save_curve_csv(out_dir, label.replace("→","to"), atk, eps_arr, ra_arr)
                png_path = save_plot_png(out_dir, f"{label} {atk.upper()} RA", [(atk.upper(), eps_arr, ra_arr)])
                print(f"Saved: {csv_path} | {png_path}")
            transfer_results[label] = {"attacks": per_attack}

    # Global summary JSON
    summary = {
        "classes": all_classes,
        "eps_from255": eps_255,
        "attacks": attacks,
        "per_model": per_model_results,
        "transfer": transfer_results if transfer_results else None
    }
    summary_path = save_summary_json(Path(args.out_dir), "summary.json", summary)
    print(f"\nSummary saved: {summary_path}")
    print("Done.")
    

if __name__ == "__main__":
    main()
