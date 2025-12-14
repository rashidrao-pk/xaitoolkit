"""XAI Method: Axis-aligned SHAP-like attributions (rectangular partitioning)

Reference / Links:
- SHAP (general): https://arxiv.org/abs/1705.07874
- This implementation is a lightweight, *SHAP-inspired* hierarchical marginal contribution scheme
  over axis-aligned image rectangles (not exact SHAP).

Idea:
1) Build a binary partition tree over rectangles (split along longer side).
2) Define f(R): probability when only rectangle R is kept (outside replaced by baseline).
3) Contribution(node) ≈ f(node) - f(parent).
4) Distribute contributions over pixels in the rectangle.

Outputs:
- heatmap in [0,1]
"""

from __future__ import annotations
import numpy as np
import torch
from PIL import Image

from xaitoolkit.utils.viz import normalize_01

@torch.no_grad()
def _prob_target(model, x, target_idx: int) -> float:
    logits = model(x)
    p = torch.softmax(logits, dim=1)[0, target_idx].item()
    return float(p)

def _make_baseline(img01: np.ndarray, mode: str = "mean") -> np.ndarray:
    if mode == "mean":
        c = img01.mean(axis=(0,1), keepdims=True)
        return np.repeat(np.repeat(c, img01.shape[0], axis=0), img01.shape[1], axis=1)
    if mode == "blur":
        pil = Image.fromarray((img01 * 255).astype(np.uint8))
        small = pil.resize((max(1, img01.shape[1]//16), max(1, img01.shape[0]//16)), resample=Image.BILINEAR)
        blur = small.resize((img01.shape[1], img01.shape[0]), resample=Image.BILINEAR)
        return np.asarray(blur).astype(np.float32) / 255.0
    raise ValueError("baseline must be 'mean' or 'blur'")

def _apply_keep_rect(img01: np.ndarray, base01: np.ndarray, rect):
    y0,y1,x0,x1 = rect
    out = base01.copy()
    out[y0:y1, x0:x1] = img01[y0:y1, x0:x1]
    return out

def _rect_split(rect):
    y0,y1,x0,x1 = rect
    h = y1 - y0
    w = x1 - x0
    if h >= w:
        ym = y0 + h//2
        return (y0, ym, x0, x1), (ym, y1, x0, x1)
    xm = x0 + w//2
    return (y0, y1, x0, xm), (y0, y1, xm, x1)

def _build_tree(H: int, W: int, max_depth: int, min_size: int):
    nodes = []
    def rec(rect, depth, parent):
        nid = len(nodes)
        node = {"id": nid, "rect": rect, "left": None, "right": None, "depth": depth, "parent": parent}
        nodes.append(node)
        y0,y1,x0,x1 = rect
        if depth >= max_depth or (y1-y0) <= min_size or (x1-x0) <= min_size:
            return nid
        r1,r2 = _rect_split(rect)
        node["left"] = rec(r1, depth+1, nid)
        node["right"] = rec(r2, depth+1, nid)
        return nid
    rec((0,H,0,W), 0, None)
    return nodes

def explain_shap_axis_aligned(
    model,
    pil_resized,
    target_idx: int,
    device: torch.device,
    preprocess,
    baseline: str = "mean",
    max_depth: int = 6,
    min_size: int = 16,
    **kwargs
) -> np.ndarray:
    img01 = np.asarray(pil_resized).astype(np.float32) / 255.0
    H, W = img01.shape[:2]
    base01 = _make_baseline(img01, mode=baseline)

    # probability on full baseline ("empty")
    pil_empty = Image.fromarray((base01 * 255).astype(np.uint8))
    x_empty = preprocess(pil_empty).unsqueeze(0).to(device)
    f_empty = _prob_target(model, x_empty, target_idx)

    nodes = _build_tree(H, W, max_depth=max_depth, min_size=min_size)

    f_node = {}
    for n in nodes:
        masked = _apply_keep_rect(img01, base01, n["rect"])
        pil_m = Image.fromarray((masked * 255).astype(np.uint8))
        xm = preprocess(pil_m).unsqueeze(0).to(device)
        f_node[n["id"]] = _prob_target(model, xm, target_idx)

    contrib = np.zeros(len(nodes), dtype=np.float32)
    for n in nodes:
        pid = n["parent"]
        if pid is None:
            contrib[n["id"]] = f_node[n["id"]] - f_empty
        else:
            contrib[n["id"]] = f_node[n["id"]] - f_node[pid]

    heat = np.zeros((H, W), dtype=np.float32)
    for n in nodes:
        y0,y1,x0,x1 = n["rect"]
        area = max(1, (y1-y0)*(x1-x0))
        heat[y0:y1, x0:x1] += contrib[n["id"]] / area

    return normalize_01(heat)
