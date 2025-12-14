"""xaitoolkit.loaders.model_loader

Model loading utilities.

Reference / Links:
- timm models: https://github.com/huggingface/pytorch-image-models
- torchvision models: https://pytorch.org/vision/stable/models.html

Supported model specs:
- timm:<model_name>
- tv:<model_name>
- ckpt:<path_to_pt>  (requires --arch timm:<name> or tv:<name>)
- py:<file.py>:<factory_fn>  (factory returns torch.nn.Module)
"""

from __future__ import annotations
import importlib.util
import os
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LoadedModel:
    model: torch.nn.Module
    source: str
    num_classes: Optional[int] = None


def _load_py_factory(py_spec: str) -> torch.nn.Module:
    # py:<file.py>:<fn>
    _, rest = py_spec.split("py:", 1)
    file_path, fn_name = rest.split(":", 1)
    file_path = os.path.abspath(file_path)

    spec = importlib.util.spec_from_file_location("user_model_module", file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    fn = getattr(module, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Factory function '{fn_name}' not found in {file_path}")
    m = fn()
    if not isinstance(m, torch.nn.Module):
        raise RuntimeError("Factory function must return torch.nn.Module")
    return m


def load_model(model_spec: str, device: torch.device, arch: Optional[str] = None) -> LoadedModel:
    model_spec = model_spec.strip()

    if model_spec.startswith("timm:"):
        import timm
        name = model_spec.split("timm:", 1)[1]
        m = timm.create_model(name, pretrained=True)
        m.eval().to(device)
        return LoadedModel(model=m, source=model_spec, num_classes=getattr(m, "num_classes", None))

    if model_spec.startswith("tv:"):
        import torchvision.models as tvm
        name = model_spec.split("tv:", 1)[1]
        if not hasattr(tvm, name):
            raise ValueError(f"torchvision has no model '{name}'")
        m = getattr(tvm, name)(weights="DEFAULT")
        m.eval().to(device)
        return LoadedModel(model=m, source=model_spec)

    if model_spec.startswith("ckpt:"):
        if arch is None:
            raise ValueError("For ckpt:<path>, pass --arch timm:<name> or tv:<name>")
        ckpt_path = os.path.abspath(model_spec.split("ckpt:", 1)[1])

        base = load_model(arch, device).model
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        if not isinstance(state, dict):
            raise RuntimeError("Checkpoint must be a state_dict (or contain one under 'state_dict').")

        base.load_state_dict(state, strict=False)
        base.eval().to(device)
        return LoadedModel(model=base, source=model_spec)

    if model_spec.startswith("py:"):
        m = _load_py_factory(model_spec)
        m.eval().to(device)
        return LoadedModel(model=m, source=model_spec)

    raise ValueError("Unknown model spec. Use timm: | tv: | ckpt: | py:")
