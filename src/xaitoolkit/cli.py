"""xaitoolkit.cli

Command line interface.

Examples:
  xai-explain --image assets/cat_dog.jpg --model tv:resnet50 --methods saliency smoothgrad ig
  xai-explain --image assets/flamingo.jpg --model timm:swin_tiny_patch4_window7_224 --methods rise lime_strat shap_axis
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from xaitoolkit.loaders.model_loader import load_model
from xaitoolkit.loaders.preprocess import default_imagenet_transform, timm_transform_if_available
from xaitoolkit.utils.viz import save_img, overlay_heatmap

from xaitoolkit.methods.saliency import explain_saliency
from xaitoolkit.methods.smoothgrad import explain_smoothgrad
from xaitoolkit.methods.ig import explain_ig
from xaitoolkit.methods.rise import explain_rise
from xaitoolkit.methods.occlusion import explain_occlusion
from xaitoolkit.methods.lime_stratified import explain_lime_stratified
from xaitoolkit.methods.shap_axis_aligned import explain_shap_axis_aligned


METHODS = {
    "saliency": explain_saliency,
    "smoothgrad": explain_smoothgrad,
    "ig": explain_ig,
    "rise": explain_rise,
    "occlusion": explain_occlusion,
    "lime_strat": explain_lime_stratified,
    "shap_axis": explain_shap_axis_aligned,
}

PIXEL_SPACE_METHODS = {"lime_strat", "shap_axis"}  # require preprocess + PIL-resized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", required=True, help="timm:<name> | tv:<name> | ckpt:<path> | py:<file.py>:<fn>")
    ap.add_argument("--arch", default=None, help="Required for ckpt:<path> (timm:<name> or tv:<name>)")
    ap.add_argument("--methods", nargs="+", default=["saliency"])
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--target", type=int, default=None, help="Class index. Default = predicted class")
    args = ap.parse_args()

    device = torch.device(args.device)

    loaded = load_model(args.model, device=device, arch=args.arch)
    model = loaded.model

    preprocess = timm_transform_if_available(model) or default_imagenet_transform(args.image_size)

    pil = Image.open(args.image).convert("RGB")
    x = preprocess(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_prob = float(probs[0, pred_idx].item())

    target = pred_idx if args.target is None else args.target
    print(f"Model: {loaded.source}")
    print(f"Pred idx: {pred_idx} | prob: {pred_prob:.4f} | target used: {target}")

    os.makedirs(args.outdir, exist_ok=True)

    # Save resized image for overlays (match x spatial size)
    H, W = int(x.shape[-2]), int(x.shape[-1])
    pil_r = pil.resize((W, H))
    img01 = np.asarray(pil_r).astype(np.float32) / 255.0
    save_img(os.path.join(args.outdir, "original.png"), img01)

    for m in args.methods:
        if m not in METHODS:
            raise ValueError(f"Unknown method '{m}'. Available: {list(METHODS.keys())}")

        fn = METHODS[m]
        if m in PIXEL_SPACE_METHODS:
            heat = fn(model=model, pil_resized=pil_r, target_idx=target, device=device, preprocess=preprocess)
        else:
            heat = fn(model=model, x=x, target_idx=target, device=device)

        save_img(os.path.join(args.outdir, f"{m}_heatmap.png"), heat, cmap="jet")
        save_img(os.path.join(args.outdir, f"{m}_overlay.png"), overlay_heatmap(img01, heat))
        print(f"Saved: {m}_heatmap.png, {m}_overlay.png")
