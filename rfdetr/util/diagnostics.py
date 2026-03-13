# ------------------------------------------------------------------------
# Spectral-DETR
# GitHub: https://github.com/songyuexin666-wq/Sprectral-DETR  (TODO: update link)
# ------------------------------------------------------------------------

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _to_serializable(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


class DiagnosticsWriter:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self._opened = False

    def write_scalars(self, data: Dict[str, Any]):
        serializable = {k: _to_serializable(v) for k, v in data.items()}
        serializable["timestamp"] = datetime.now().isoformat()
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    def write_json(self, name: str, data: Dict[str, Any]):
        path = self.output_dir / f"{name}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def save_grayscale(self, name: str, image: np.ndarray):
        if plt is None:
            return
        path = self.output_dir / f"{name}.png"
        plt.figure(figsize=(4, 4))
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def save_scatter(self, name: str, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str):
        if plt is None:
            return
        path = self.output_dir / f"{name}.png"
        plt.figure(figsize=(4, 4))
        plt.scatter(x, y, s=6, alpha=0.5)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def save_hist_2(self, name: str, a: np.ndarray, b: np.ndarray, label_a: str, label_b: str, xlabel: str):
        if plt is None:
            return
        path = self.output_dir / f"{name}.png"
        plt.figure(figsize=(4, 4))
        bins = 40
        plt.hist(a, bins=bins, alpha=0.6, label=label_a)
        plt.hist(b, bins=bins, alpha=0.6, label=label_b)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
