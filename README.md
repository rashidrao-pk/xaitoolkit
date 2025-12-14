# XAIToolkit — Explain any PyTorch CNN in minutes

A lightweight, **batteries-included** toolkit to explain image-classification models (your own CNN, torchvision models, or timm models) using:
- **Gradient-based XAI**: Saliency, SmoothGrad, Integrated Gradients
- **Model-agnostic XAI**: RISE, Occlusion
- **Local surrogate XAI**: **LIME-Stratified (superpixels)** (stable neighborhood sampling)
- **Region / tree-style XAI**: Axis-aligned SHAP-like attributions (rectangle partitioning)

This repo starts with a teaching-first notebook and also ships as a small **Python package** + **CLI**.

## Quick start

### 1) Install (editable for development)

```bash
pip install -e .
```

### 2) Run the notebook

Open:

- `notebooks/01_grad_based_xai.ipynb`

It loads a CNN, predicts Top-5 classes, and visualizes three gradient-based explainers in a 2×3 figure.

### 3) Explain an image from the CLI

Torchvision model:

```bash
xai-explain --image assets/cat_dog.jpg --model tv:resnet50 --methods saliency smoothgrad ig --outdir outputs
```

timm model (pretrained):

```bash
xai-explain --image assets/flamingo.jpg --model timm:swin_tiny_patch4_window7_224 --methods rise lime_strat shap_axis --outdir outputs
```

Your checkpoint + architecture:

```bash
xai-explain --image assets/cat_dog.jpg --model ckpt:checkpoints/best.pt --arch tv:resnet50 --methods rise occlusion --outdir outputs
```

## What gets saved

For each method:

- `outputs/<method>_heatmap.png`
- `outputs/<method>_overlay.png`

Plus `outputs/original.png`.

## Methods

### Gradient-based
- `saliency`: ∂logit/∂input (absolute, channel-mean)
- `smoothgrad`: noise-averaged saliency
- `ig`: Integrated Gradients (manual implementation, no Captum)

### Model-agnostic
- `rise`: Randomized Input Sampling for Explanation (RISE)
- `occlusion`: Sliding-window occlusion sensitivity

### Surrogate / region-based
- `lime_strat`: LIME Image using **stratified sampling** of the neighborhood (bins on model output)
- `shap_axis`: Axis-aligned SHAP-like attributions using hierarchical rectangle splits

Each method file includes references and canonical links at the top.

## Bring your own model

You can load models in four ways:

- `tv:<name>` — torchvision, e.g. `tv:resnet50`
- `timm:<name>` — timm pretrained models
- `ckpt:<path>` + `--arch tv:<name>|timm:<name>` — load checkpoint into a known architecture
- `py:<file.py>:<factory_fn>` — load a custom model factory that returns `torch.nn.Module`

## Project layout

- `src/xaitoolkit/` — package code
- `scripts/` — small wrappers (CLI lives here)
- `notebooks/` — teaching notebook(s)
- `assets/` — demo images
- `outputs/` — generated artifacts (gitignored)

## Citation & credits

- ResNet: https://arxiv.org/abs/1512.03385  
- Integrated Gradients: https://arxiv.org/abs/1703.01365  
- SmoothGrad: https://arxiv.org/abs/1706.03825  
- Grad-CAM: https://arxiv.org/abs/1610.02391  
- RISE: https://arxiv.org/abs/1806.07421  

<!-- ## RoadMaps -->
 