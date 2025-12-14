"""XAI Method: RISE (Randomized Input Sampling for Explanation)

Reference / Links:
- Paper: https://arxiv.org/abs/1806.07421
- Official project page (paper): https://arxiv.org/abs/1806.07421

Outputs:
- heatmap in [0,1]
"""

from __future__ import annotations
import numpy as np
import torch

from xaitoolkit.utils.viz import normalize_01

@torch.no_grad()
def _model_prob(model, x_batch, target_idx: int) -> torch.Tensor:
    logits = model(x_batch)
    prob = torch.softmax(logits, dim=1)[:, target_idx]
    return prob

def explain_rise(
    model,
    x,
    target_idx: int,
    device: torch.device,
    num_masks: int = 800,
    grid: int = 8,
    p1: float = 0.5,
    batch_size: int = 64,
    **kwargs
) -> np.ndarray:
    # x: [1,3,H,W]
    x = x.clone().detach().to(device)
    _, _, H, W = x.shape
    s = int(grid)

    rng = np.random.default_rng(kwargs.get("seed", None))

    grids = (rng.random((num_masks, s, s)) < p1).astype(np.float32)

    cell_h = int(np.ceil(H / s))
    cell_w = int(np.ceil(W / s))
    up_h = cell_h * s
    up_w = cell_w * s

    masks = np.zeros((num_masks, H, W), dtype=np.float32)
    for i in range(num_masks):
        g = grids[i]
        g_up = np.kron(g, np.ones((cell_h, cell_w), dtype=np.float32))  # [up_h, up_w]

        # pad so shift+crop always yields HxW
        g_up = np.pad(g_up, ((0, cell_h), (0, cell_w)), mode="wrap")

        shift_y = rng.integers(0, cell_h)
        shift_x = rng.integers(0, cell_w)
        masks[i] = g_up[shift_y:shift_y + H, shift_x:shift_x + W]

    masks_t = torch.from_numpy(masks).to(device).unsqueeze(1)  # [N,1,H,W]

    scores = []
    with torch.no_grad():
        for i in range(0, num_masks, batch_size):
            m = masks_t[i:i + batch_size]
            xm = x * m
            scores.append(_model_prob(model, xm, target_idx).detach())
    scores = torch.cat(scores, dim=0)  # [N]

    sal = (scores.view(-1, 1, 1, 1) * masks_t).sum(dim=0)[0]  # [H,W]
    return normalize_01(sal.detach().cpu().numpy())
