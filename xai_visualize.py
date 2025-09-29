#!/usr/bin/env python3
"""
xai_visualize.py — Saliency & Grad-CAM(++) for models saved as .pth (state_dict).

What it does
------------
- Loads model weights from ./models and metadata (<model>_meta.json)
- Runs single-image or folder (recursive) inference
- Generates:
    * Vanilla Grad-based Saliency (|dLogit/dInput|)
    * Grad-CAM (and optionally Grad-CAM++) on a chosen convolutional layer
- Saves: heatmap PNG and overlay PNG per image and per model

Notes
-----
- CNNs (ResNet/ShuffleNet/RegNet): Grad-CAM is straightforward; default target layers provided.
- Swin-T: transformer backbone — Grad-CAM is not directly applicable. Script defaults to Saliency;
  use --target-layer to pick a convolutional layer (e.g., patch embedding) if you really want Grad-CAM-like maps.
- For Grad-CAM++ set --gradcam_plus_plus to enable the alternative weighting.

Usage examples
--------------
# Single image, Saliency + Grad-CAM (default), for ResNet-50
python xai_visualize.py --image path/to/img.jpg --model resnet50

# Folder of images (recursively), Grad-CAM only, RegNetY-8GF, custom target layer
python xai_visualize.py --folder ./samples --model regnety_8gf --method gradcam --target-layer stem.conv

# Saliency only for Swin-T
python xai_visualize.py --image path/to/img.jpg --model swin_t --method saliency

# Enable Grad-CAM++ and write outputs to a custom folder
python xai_visualize.py --image path/to/img.jpg --model resnet50 --gradcam-plus-plus --out ./xai_out
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models

# Optional timm for RegNetY-8GF and Swin-T
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False


# ------------------------- Model factory (your 4 models) -------------------------
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
        raise ValueError(f"{name} requires 'timm' (pip install timm).")
    timm_ids = {
        "regnety_8gf": "regnety_008.pycls_in1k",
        "swin_t": "swin_tiny_patch4_window7_224.ms_in22k",
    }
    if n not in timm_ids:
        raise ValueError("Supported: resnet50, shufflenet_v2_x1_0, regnety_8gf, swin_t")
    return timm.create_model(timm_ids[n], pretrained=False, num_classes=num_classes)


# ------------------------- Preprocessing -------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size*1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def load_image_tensor(path: str, img_size: int, device: torch.device) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    t = build_transform(img_size)(img).unsqueeze(0).to(device)  # [1,C,H,W]
    return t

def tensor_to_uint8_image(t: torch.Tensor) -> np.ndarray:
    # t shape: [1,3,H,W] normalized — convert back approximately for overlay background
    x = t.detach().cpu().clone()
    for c in range(3):
        x[0, c] = x[0, c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    x = x.clamp(0, 1)
    x = (x[0].permute(1,2,0).numpy()*255.0).astype(np.uint8)
    return x


# ------------------------- Grad-CAM machinery -------------------------
class FeatureHook:
    """Stores activations and gradients for a chosen module."""
    def __init__(self, module: nn.Module):
        self.fmap = None
        self.grad = None
        self.hook_f = module.register_forward_hook(self._forward)
        self.hook_b = module.register_full_backward_hook(self._backward)

    def _forward(self, module, inp, out):
        self.fmap = out.detach()

    def _backward(self, module, grad_in, grad_out):
        # grad_out is a tuple; take grads w.r.t. output feature map
        self.grad = grad_out[0].detach()

    def close(self):
        self.hook_f.remove()
        self.hook_b.remove()

def find_default_target_layer(model: nn.Module, model_name: str) -> Optional[str]:
    """
    Heuristics for a good conv layer to use for Grad-CAM for each architecture.
    Returns a 'dotted path' to the module, e.g. 'layer4.2.conv3' for ResNet-50.
    """
    m = model_name.lower()
    if m == "resnet50":
        return "layer4.2.conv3"     # last conv in the final bottleneck
    if m == "shufflenet_v2_x1_0":
        return "conv5"              # final conv before classifier head
    if m == "regnety_8gf":
        return "stem.conv"          # or 's4' blocks; depends on timm variant; stem.conv is safe
    if m == "swin_t":
        # Swin is transformer; no spatial conv maps at the end. Use patch embedding if needed.
        return None
    return None

def get_module_by_name(model: nn.Module, dotted: str) -> nn.Module:
    obj = model
    for attr in dotted.split("."):
        if attr.isdigit():
            obj = obj[int(attr)]
        else:
            obj = getattr(obj, attr)
    return obj

def gradcam_heatmap(model: nn.Module, logits: torch.Tensor, class_idx: int,
                    fmap: torch.Tensor, grad: torch.Tensor,
                    plus_plus: bool = False) -> torch.Tensor:
    """
    Compute Grad-CAM or Grad-CAM++ heatmap (unnormalized), shape [H, W].
    """
    # fmap: [B, C, H, W], grad: [B, C, H, W], here B=1
    A = fmap[0]      # [C,H,W]
    dY = grad[0]     # [C,H,W]

    if not plus_plus:
        # Standard Grad-CAM: weights = GAP over spatial dims of grads
        weights = dY.mean(dim=(1,2))                     # [C]
        cam = (weights.view(-1,1,1) * A).sum(dim=0)      # [H,W]
    else:
        # Grad-CAM++ (per Chattopadhyay et al.)
        # alpha_k = sum_ij (d2Y/dA^2) / (2 * sum_ij (d2Y/dA^2) + sum_ij A * d3Y/dA^3)
        # In practice we approximate using ReLU on grads and normalize channels
        relu_dY = F.relu(dY)
        sum_relu_dY = relu_dY.sum(dim=(1,2)) + 1e-6      # [C]
        weights = (relu_dY / sum_relu_dY.view(-1,1,1)).sum(dim=(1,2))  # [C]
        cam = (weights.view(-1,1,1) * A).sum(dim=0)      # [H,W]

    cam = F.relu(cam)                                    # keep positive contributions
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam  # [H,W] in [0,1]


# ------------------------- Saliency -------------------------
def saliency_map(input_tensor: torch.Tensor, model: nn.Module, class_idx: int) -> torch.Tensor:
    """
    Vanilla saliency: |dLogit_c / dInput| aggregated over channels (max/abs).
    Returns [H,W] normalized to [0,1].
    """
    x = input_tensor.clone().detach().requires_grad_(True)
    logits = model(x)
    score = logits[0, class_idx]
    model.zero_grad(set_to_none=True)
    score.backward()
    grad = x.grad.detach()[0]              # [3,H,W]
    # channel-aggregation: take absolute and max over channels
    grad = grad.abs().max(dim=0)[0]        # [H,W]
    grad = grad - grad.min()
    denom = grad.max() - grad.min() + 1e-8
    grad = grad / denom
    return grad


# ------------------------- Utilities -------------------------
def overlay_heatmap_on_image(img_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Blend a heatmap (0..1) on top of an RGB image (uint8) using a simple colormap.
    """
    import matplotlib.cm as cm
    H, W, _ = img_uint8.shape
    heat = (heatmap * 255.0).astype(np.uint8)
    heat_color = cm.jet(heat)[:, :, :3]    # [H,W,3] in 0..1
    heat_color = (heat_color * 255.0).astype(np.uint8)
    overlay = (alpha * heat_color + (1 - alpha) * img_uint8).astype(np.uint8)
    return overlay

def list_images_recursive(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(str(Path(root) / f))
    return sorted(paths)


# ------------------------- Main -------------------------
def main():
    ap = argparse.ArgumentParser(description="Saliency & Grad-CAM(++) visualizations for saved .pth models")
    ap.add_argument("--image", type=str, default=None, help="Path to a single image")
    ap.add_argument("--folder", type=str, default=None, help="Path to a folder (recursively scans images)")
    ap.add_argument("--model", type=str, required=True,
                    choices=["resnet50", "regnety_8gf", "shufflenet_v2_x1_0", "swin_t"],
                    help="Which model to load from ./models/<model>.pth")
    ap.add_argument("--models_dir", type=str, default="./models", help="Folder with <model>.pth and <model>_meta.json")
    ap.add_argument("--method", type=str, default="both", choices=["saliency", "gradcam", "both"],
                    help="Which visualization(s) to produce")
    ap.add_argument("--gradcam-plus-plus", action="store_true", help="Use Grad-CAM++ weighting instead of vanilla")
    ap.add_argument("--target-layer", type=str, default=None,
                    help="Dotted path to the target conv layer (e.g., 'layer4.2.conv3'). If omitted, uses a sensible default per model; Swin-T has no default.")
    ap.add_argument("--topk", type=int, default=1, help="Visualize with respect to top-k predicted classes (default: only top-1)")
    ap.add_argument("--out", type=str, default="./xai_outputs", help="Output folder")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if (args.image is None) == (args.folder is None):
        raise ValueError("Provide exactly one of --image or --folder.")

    device = torch.device(args.device)

    # Load metadata
    meta_path = Path(args.models_dir) / f"{args.model}_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta JSON: {meta_path}. Train script should have created it.")
    meta = json_load(meta_path)
    classes = meta["classes"]
    img_size = int(meta.get("img_size", 224))
    num_classes = len(classes)

    # Build & load model weights
    model = build_model(args.model, num_classes=num_classes).to(device)
    weights_path = Path(args.models_dir) / f"{args.model}.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights file: {weights_path}")
    sd = torch.load(weights_path, map_location=device)
    try:
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"[WARN] strict load failed: {e}; trying strict=False")
        model.load_state_dict(sd, strict=False)
    model.eval()

    # Target layer for Grad-CAM
    target_layer_name = args.target_layer or find_default_target_layer(model, args.model)
    if args.method in ("gradcam", "both") and target_layer_name is None:
        print("[INFO] No default Grad-CAM layer for this model (likely Swin-T). "
              "Saliency will work; for Grad-CAM, pass --target-layer (e.g. 'patch_embed.proj').")

    # Collect images
    if args.image:
        image_paths = [args.image]
    else:
        image_paths = list_images_recursive(args.folder)
        if not image_paths:
            raise FileNotFoundError("No images found in the provided folder.")

    out_root = Path(args.out) / args.model
    out_root.mkdir(parents=True, exist_ok=True)

    # Hook if Grad-CAM is requested and we have a target layer
    hook = None
    target_module = None
    if args.method in ("gradcam", "both") and target_layer_name:
        try:
            target_module = get_module_by_name(model, target_layer_name)
            hook = FeatureHook(target_module)
        except Exception as e:
            print(f"[WARN] Could not register Grad-CAM hook for '{target_layer_name}': {e}")
            target_module = None

    # Process images
    for img_path in image_paths:
        x = load_image_tensor(img_path, img_size, device)
        img_uint8 = tensor_to_uint8_image(x)

        with torch.no_grad():
            logits = model(x)
            probs = logits.softmax(dim=1)[0].detach().cpu().numpy()
        top_indices = np.argsort(probs)[::-1][:max(1, args.topk)]

        base_name = Path(img_path).stem

        # For each selected class index, make visualizations
        for rank, cls_idx in enumerate(top_indices, start=1):
            cls_name = classes[cls_idx]

            # Saliency
            if args.method in ("saliency", "both"):
                sal = saliency_map(x, model, cls_idx).cpu().numpy()
                sal_overlay = overlay_heatmap_on_image(img_uint8, sal, alpha=0.55)
                out_sal = out_root / f"{base_name}_saliency_top{rank}_{cls_name}.png"
                out_sal_overlay = out_root / f"{base_name}_saliency_top{rank}_{cls_name}_overlay.png"
                Image.fromarray((sal*255).astype(np.uint8)).save(out_sal)
                Image.fromarray(sal_overlay).save(out_sal_overlay)

            # Grad-CAM / Grad-CAM++
            if args.method in ("gradcam", "both") and target_module is not None and hook is not None:
                # Forward pass already done; we need a backward for target class
                model.zero_grad(set_to_none=True)
                # Re-run forward to ensure hooks capture this pass
                logits = model(x)
                score = logits[0, cls_idx]
                score.backward(retain_graph=True)

                fmap = hook.fmap
                grad = hook.grad
                if fmap is None or grad is None:
                    print(f"[WARN] Missing hooks for Grad-CAM on {img_path} — skipping.")
                else:
                    cam = gradcam_heatmap(model, logits, cls_idx, fmap, grad, plus_plus=args.gradcam_plus_plus)
                    cam_np = cam.detach().cpu().numpy()
                    # Resize CAM to image size
                    cam_t = torch.from_numpy(cam_np).unsqueeze(0).unsqueeze(0).float()  # [1,1,h,w]
                    cam_up = F.interpolate(cam_t, size=img_uint8.shape[:2], mode="bilinear", align_corners=False)[0,0].clamp(0,1)
                    cam_up_np = cam_up.numpy()
                    cam_overlay = overlay_heatmap_on_image(img_uint8, cam_up_np, alpha=0.55)
                    tag = "gradcampp" if args.gradcam_plus_plus else "gradcam"
                    out_cam = out_root / f"{base_name}_{tag}_top{rank}_{cls_name}.png"
                    out_cam_overlay = out_root / f"{base_name}_{tag}_top{rank}_{cls_name}_overlay.png"
                    Image.fromarray((cam_up_np*255).astype(np.uint8)).save(out_cam)
                    Image.fromarray(cam_overlay).save(out_cam_overlay)

        print(f"[OK] {img_path} → {out_root}")

    if hook is not None:
        hook.close()


    import json
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
