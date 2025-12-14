"""XAI Method: Vanilla Saliency (input gradients)

Reference / Links:
- Common baseline used widely in interpretability literature.
- (Background) Simonyan et al., "Deep Inside Convolutional Networks" (2013): https://arxiv.org/abs/1312.6034

Outputs:
- heatmap in [0,1]
"""

from __future__ import annotations
import numpy as np
import torch

from xaitoolkit.utils.viz import normalize_01

def explain_saliency(model, x, target_idx: int, device: torch.device, **kwargs) -> np.ndarray:
    # x: [1,3,H,W] preprocessed
    x = x.clone().detach().to(device).requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits[0, target_idx]  # logit is more stable than prob
    score.backward()
    grad = x.grad.detach()[0].abs().mean(dim=0).cpu().numpy()  # [H,W]
    return normalize_01(grad)
