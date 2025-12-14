"""XAI Method: SmoothGrad

Reference / Links:
- Paper: https://arxiv.org/abs/1706.03825

Idea:
Average saliency maps over noisy copies of the input to reduce visual noise.

Outputs:
- heatmap in [0,1]
"""

from __future__ import annotations
import numpy as np
import torch

from xaitoolkit.utils.viz import normalize_01

def explain_smoothgrad(
    model,
    x,
    target_idx: int,
    device: torch.device,
    n_samples: int = 30,
    noise_sigma: float = 0.12,
    **kwargs
) -> np.ndarray:
    x0 = x.clone().detach().to(device)
    grads = []
    for _ in range(n_samples):
        noise = torch.randn_like(x0) * noise_sigma
        xn = (x0 + noise).clone().detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        logits = model(xn)
        score = logits[0, target_idx]
        score.backward()
        g = xn.grad.detach()[0].abs().mean(dim=0)  # [H,W]
        grads.append(g)
    g_mean = torch.stack(grads, dim=0).mean(dim=0).cpu().numpy()
    return normalize_01(g_mean)
