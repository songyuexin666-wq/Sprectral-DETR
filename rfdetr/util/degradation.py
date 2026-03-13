# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

from typing import Dict, Tuple, List
import torch
import torch.nn.functional as F


_LAPLACIAN_KERNEL = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]],
    dtype=torch.float32
).view(1, 1, 3, 3)


def _denormalize(images: torch.Tensor):
    device = images.device
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return (images * std + mean).clamp(0, 1)


def compute_degradation_scores(images: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    images: normalized tensor [B,3,H,W]
    returns brightness/contrast/blur score per image
    """
    images = _denormalize(images)
    gray = (0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]).unsqueeze(1)

    brightness = gray.mean(dim=(2, 3))
    contrast = gray.std(dim=(2, 3))

    kernel = _LAPLACIAN_KERNEL.to(images.device)
    lap = F.conv2d(gray, kernel, padding=1)
    blur = lap.var(dim=(2, 3))

    return {
        "brightness": brightness.squeeze(1),
        "contrast": contrast.squeeze(1),
        "blur": blur.squeeze(1),
    }


def bucketize(values: torch.Tensor, thresholds: List[float], labels: List[str]) -> List[str]:
    """
    thresholds: sorted ascending, length = len(labels)-1
    """
    buckets = []
    for v in values.detach().cpu().tolist():
        idx = 0
        while idx < len(thresholds) and v >= thresholds[idx]:
            idx += 1
        buckets.append(labels[idx])
    return buckets
