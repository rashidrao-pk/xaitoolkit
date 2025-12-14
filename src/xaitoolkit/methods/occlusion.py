"""XAI Method: Occlusion sensitivity (sliding-window masking)

Reference / Links:
- Often used as a model-agnostic baseline in interpretability work.
- Related idea appears in: Zeiler & Fergus (2014) "Visualizing and Understanding ConvNets" https://arxiv.org/abs/1311.2901

Idea:
Mask a patch and measure drop in target probability.

Outputs:
- heatmap in [0,1]
"""

from __future__ import annotations
import numpy as np
import torch

from xaitoolkit.utils.viz import normalize_01

@torch.no_grad()
def explain_occlusion(
    model,
    x,
    target_idx: int,
    device: torch.device,
    patch: int = 32,
    stride: int = 16,
    baseline: str = "zero",
    **kwargs
) -> np.ndarray:
    x = x.clone().detach().to(device)
    B, C, H, W = x.shape
    assert B == 1, "Occlusion expects a single image (B=1)."

    logits = model(x)
    p0 = torch.softmax(logits, dim=1)[0, target_idx].item()

    heat = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    if baseline == "mean":
        fill = x.mean().item()
    else:
        fill = 0.0

    for y in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y + patch, H)
            x1 = min(x0 + patch, W)

            x_occ = x.clone()
            x_occ[:, :, y:y1, x0:x1] = fill

            p = torch.softmax(model(x_occ), dim=1)[0, target_idx].item()
            drop = max(0.0, p0 - p)

            heat[y:y1, x0:x1] += drop
            counts[y:y1, x0:x1] += 1.0

    heat = heat / (counts + 1e-8)
    return normalize_01(heat)
