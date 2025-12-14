import os
import numpy as np
import matplotlib.pyplot as plt

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def normalize_01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = arr - arr.min()
    return arr / (arr.max() + 1e-8)

def overlay_heatmap(img01: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    hm = cmap(heat01)[..., :3]
    out = (1 - alpha) * img01 + alpha * hm
    return np.clip(out, 0, 1)

def save_img(path: str, arr01: np.ndarray, cmap: str = None) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    if cmap is None:
        plt.imsave(path, arr01)
    else:
        plt.imsave(path, arr01, cmap=cmap)
