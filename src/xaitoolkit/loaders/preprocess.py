"""xaitoolkit.loaders.preprocess

Preprocessing helpers.

Reference / Links:
- torchvision transforms: https://pytorch.org/vision/stable/transforms.html
- timm data config: https://huggingface.co/docs/timm

Goal:
- Provide a default ImageNet-like transform
- Use timm's resolve_data_config when available
"""

from __future__ import annotations
from typing import Callable, Optional
from PIL import Image
import torch

def default_imagenet_transform(image_size: int = 224) -> Callable[[Image.Image], torch.Tensor]:
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def timm_transform_if_available(model) -> Optional[Callable]:
    try:
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        cfg = resolve_data_config({}, model=model)
        return create_transform(**cfg)
    except Exception:
        return None
