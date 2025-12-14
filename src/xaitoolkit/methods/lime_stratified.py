"""XAI Method: LIME-Stratified (Image, superpixels)

Reference / Links:
- LIME paper: https://arxiv.org/abs/1602.04938
- Your stratified neighborhood idea: (add your paper / repo link here)

Notes:
- We build a superpixel segmentation (SLIC).
- We generate perturbed samples by toggling superpixels on/off.
- We enforce *stratified sampling* over model outputs (bins on p(target)) to stabilize explanations.
- We fit a weighted Ridge surrogate model.

Outputs:
- heatmap in [0,1]
"""

from __future__ import annotations
import numpy as np
import torch
from sklearn.linear_model import Ridge
from skimage.segmentation import slic

from xaitoolkit.utils.viz import normalize_01

def _cosine_distance(Z: np.ndarray, x0: np.ndarray) -> np.ndarray:
    Z = Z.astype(np.float32)
    x0 = x0.astype(np.float32)
    Z_n = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
    x0_n = x0 / (np.linalg.norm(x0) + 1e-8)
    return 1.0 - (Z_n * x0_n[None, :]).sum(axis=1)

def _kernel(d: np.ndarray, sigma: float = 0.25) -> np.ndarray:
    return np.exp(-(d ** 2) / (sigma ** 2))

@torch.no_grad()
def _prob_target(model, x_batch: torch.Tensor, target_idx: int) -> np.ndarray:
    logits = model(x_batch)
    p = torch.softmax(logits, dim=1)[:, target_idx]
    return p.detach().cpu().numpy()

def _apply_masks(img01: np.ndarray, segments: np.ndarray, Z: np.ndarray) -> np.ndarray:
    # img01: [H,W,3] in 0..1 ; Z: [N,M]
    H, W, _ = img01.shape
    sp_ids = np.unique(segments)
    masks = [(segments == sid) for sid in sp_ids]
    mean_color = img01.mean(axis=(0, 1), keepdims=True)[0, 0]

    out = np.empty((Z.shape[0], H, W, 3), dtype=np.float32)
    for i in range(Z.shape[0]):
        im = img01.copy()
        for j in range(len(sp_ids)):
            if Z[i, j] == 0:
                im[masks[j]] = mean_color
        out[i] = im
    return out  # [N,H,W,3]

def explain_lime_stratified(
    model,
    pil_resized,
    target_idx: int,
    device: torch.device,
    preprocess,
    n_segments: int = 60,
    compactness: float = 10.0,
    n_samples: int = 1200,
    n_bins: int = 6,
    topk: int = 10,
    seed: int = 0,
    **kwargs
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    img01 = np.asarray(pil_resized).astype(np.float32) / 255.0
    H, W, _ = img01.shape

    segments = slic(img01, n_segments=n_segments, compactness=compactness, start_label=0)
    sp_ids = np.unique(segments)
    M = len(sp_ids)

    # original probability
    x0 = preprocess(pil_resized).unsqueeze(0).to(device)
    y0 = _prob_target(model, x0, target_idx)[0]

    def gen_masks(k: int) -> np.ndarray:
        return (rng.random((k, M)) < 0.5).astype(np.int32)

    warmup = min(400, max(100, n_samples // 3))
    Z_w = gen_masks(warmup)
    imgs_w = _apply_masks(img01, segments, Z_w)

    # batch eval
    y_w = []
    with torch.no_grad():
        for i in range(0, warmup, 64):
            batch = imgs_w[i:i+64]
            xb = torch.stack([preprocess(Image_from_arr01(b)) for b in batch], dim=0).to(device)
            y_w.append(_prob_target(model, xb, target_idx))
    y_w = np.concatenate(y_w, axis=0)

    # quantile edges
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_w, qs)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    def bin_id(y: float) -> int:
        b = np.searchsorted(edges, y, side="right") - 1
        return int(np.clip(b, 0, len(edges) - 2))

    per_bin = max(1, (n_samples - 1) // (len(edges) - 1))
    quotas = {b: per_bin for b in range(len(edges) - 1)}
    quotas[bin_id(float(y0))] = max(0, quotas[bin_id(float(y0))] - 1)

    Z_keep = [np.ones((M,), dtype=np.int32)]
    y_keep = [float(y0)]

    # accept from warmup
    for zi, yi in zip(Z_w, y_w):
        b = bin_id(float(yi))
        if quotas.get(b, 0) > 0:
            Z_keep.append(zi)
            y_keep.append(float(yi))
            quotas[b] -= 1
        if len(Z_keep) >= n_samples:
            break

    # top-up
    while len(Z_keep) < n_samples and sum(quotas.values()) > 0:
        Z_b = gen_masks(128)
        imgs_b = _apply_masks(img01, segments, Z_b)
        # eval
        y_b = []
        for i in range(0, imgs_b.shape[0], 64):
            batch = imgs_b[i:i+64]
            xb = torch.stack([preprocess(Image_from_arr01(b)) for b in batch], dim=0).to(device)
            y_b.append(_prob_target(model, xb, target_idx))
        y_b = np.concatenate(y_b, axis=0)

        for zi, yi in zip(Z_b, y_b):
            b = bin_id(float(yi))
            if quotas.get(b, 0) > 0:
                Z_keep.append(zi)
                y_keep.append(float(yi))
                quotas[b] -= 1
            if len(Z_keep) >= n_samples:
                break

        # fail-safe: if stuck, break and fit with what we have
        if len(Z_keep) < n_samples and sum(quotas.values()) == 0:
            break

    Z = np.stack(Z_keep, axis=0)
    y = np.array(y_keep, dtype=np.float32)

    # weights by distance to all-ones
    x0_bin = np.ones((M,), dtype=np.float32)
    d = _cosine_distance(Z, x0_bin)
    w = _kernel(d, sigma=0.25)

    reg = Ridge(alpha=1.0, fit_intercept=True)
    reg.fit(Z, y, sample_weight=w)
    coefs = reg.coef_.astype(np.float32)  # [M]

    heat = np.zeros((H, W), dtype=np.float32)
    for j, sid in enumerate(sp_ids):
        heat[segments == sid] = max(0.0, coefs[j])

    return normalize_01(heat)

# ---- tiny helper to avoid PIL import cycles ----
from PIL import Image as _PILImage
def Image_from_arr01(arr01: np.ndarray) -> _PILImage:
    return _PILImage.fromarray((np.clip(arr01,0,1) * 255).astype(np.uint8))
