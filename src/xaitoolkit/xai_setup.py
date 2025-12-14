"""
xaitoolkit.xai_setup

A small helper to load a torchvision CNN + preprocessing + class labels and expose
a "black-box" predict_proba function.

Links:
- ResNet paper: https://arxiv.org/abs/1512.03385
- Torchvision models: https://pytorch.org/vision/stable/models.html
"""

from typing import Optional, Tuple, Callable, List
import torch
import torch.nn.functional as F
from torchvision import models


def xai_setup(
    model_name: str = "resnet50",
    device: Optional[str] = None,
    eval_mode: bool = True,
):
    """
    Returns:
        model: torch.nn.Module
        preprocess: callable (PIL -> tensor)
        class_names: Optional[list[str]]
        device: torch.device
        predict_proba: callable(x)-> probs on CPU
        predict_topk: callable(x,k)-> (top_probs, top_idx, top_labels)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    model_name = model_name.lower().strip()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        preprocess = weights.transforms()
        class_names = weights.meta.get("categories", None)

    elif model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
        preprocess = weights.transforms()
        class_names = weights.meta.get("categories", None)

    elif model_name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        model = models.mobilenet_v3_large(weights=weights)
        preprocess = weights.transforms()
        class_names = weights.meta.get("categories", None)

    else:
        raise ValueError(
            f"Unsupported model_name='{model_name}'. "
            "Try: resnet50, resnet18, mobilenet_v3_large."
        )

    model = model.to(device_t)
    if eval_mode:
        model.eval()

    @torch.no_grad()
    def predict_proba(x: torch.Tensor) -> torch.Tensor:
        x = x.to(device_t)
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu()

    @torch.no_grad()
    def predict_topk(x: torch.Tensor, k: int = 5):
        probs = predict_proba(x)
        top_probs, top_idx = probs[0].topk(k)
        top_probs = top_probs.numpy()
        top_idx = top_idx.numpy()
        if class_names is None:
            top_labels = [str(i) for i in top_idx]
        else:
            top_labels = [class_names[i] for i in top_idx]
        return top_probs, top_idx, top_labels

    return model, preprocess, class_names, device_t, predict_proba, predict_topk
