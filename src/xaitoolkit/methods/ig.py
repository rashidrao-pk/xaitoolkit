"""XAI Method: Integrated Gradients (manual)

Reference / Links:
- Paper: https://arxiv.org/abs/1703.01365

Notes:
- This is a manual IG implementation (no Captum dependency).
- Baseline defaults to all-zeros in input space (after preprocessing).

Outputs:
- heatmap in [0,1]
"""

from __future__ import annotations
import numpy as np
import torch

from xaitoolkit.utils.viz import normalize_01

def explain_ig(
    model,
    x,
    target_idx: int,
    device: torch.device,
    steps: int = 50,
    baseline: torch.Tensor = None,
    **kwargs
) -> np.ndarray:
    x = x.clone().detach().to(device)
    if baseline is None:
        baseline = torch.zeros_like(x)

    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1, 1, 1)
    x_exp = baseline + alphas * (x - baseline)            # [steps,3,H,W]
    x_exp = x_exp.requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(x_exp)
    scores = logits[:, target_idx].sum()
    scores.backward()

    grads = x_exp.grad.detach()                           # [steps,3,H,W]
    avg_grads = grads.mean(dim=0, keepdim=True)          # [1,3,H,W]
    ig = (x - baseline) * avg_grads                       # [1,3,H,W]
    heat = ig.detach()[0].abs().mean(dim=0).cpu().numpy() # [H,W]
    return normalize_01(heat)
