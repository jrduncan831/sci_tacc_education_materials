"""Image-to-(x, y, value) dataset for neural-field training.

Loads a 2D image .npy as flat coordinate/value tensors on device, split
into train and validation pixel subsets. Small enough to fit entirely in
GPU memory — we skip DataLoader and batch by indexing into pre-loaded
tensors.

Training on a small fraction of pixels demonstrates the surrogate-model
power of neural fields: the network learns a continuous function that can
be queried at any (x, y), including positions it never saw during training.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

DEFAULT_PATH = "/scratch/projects/tacc/sci-training/cosmicai_data/cmb/cmb_1k.npy"


class ImageDataset(NamedTuple):
    train_coords: torch.Tensor   # (N_train, 2) float32, in [-1, 1]^2
    train_values: torch.Tensor   # (N_train, 1) float32, in [-1, 1]
    val_coords:   torch.Tensor   # (N_val,   2)
    val_values:   torch.Tensor   # (N_val,   1)
    image_shape:  tuple[int, int]  # (h, w) of the original image


def load_image_dataset(
    path: str | Path = DEFAULT_PATH,
    device: str | torch.device = "cuda",
    train_fraction: float = 0.1,
    seed: int = 0,
) -> ImageDataset:
    """Load a 2D image .npy and split into train/validation pixel subsets.

    Each pixel becomes one example: input = (x, y), target = value.
    Coords are normalized to [-1, 1] x [-1, 1] (aspect ratio not preserved).
    Values are min-max normalized to [-1, 1] (the range SIREN's init scheme
    is calibrated for; harmless for ReLU).

    train_fraction: fraction of pixels used for training (the rest go to
                    validation). Small values (0.01-0.05) make the surrogate-
                    model point vivid: a neural field can learn a continuous
                    function from a sparse sample and predict the full image.
    seed: RNG seed for the train/val split, for reproducibility.
    """
    img = np.load(path).astype(np.float32)
    h, w = img.shape
    img = 2 * (img - img.min()) / (img.max() - img.min()) - 1

    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing="ij",
    )
    coords = torch.stack([xs, ys], dim=-1).reshape(-1, 2)
    values = torch.from_numpy(img).reshape(-1, 1)

    n_total = h * w
    n_train = int(train_fraction * n_total)
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    return ImageDataset(
        train_coords=coords[train_idx].to(device),
        train_values=values[train_idx].to(device),
        val_coords=coords[val_idx].to(device),
        val_values=values[val_idx].to(device),
        image_shape=(h, w),
    )


def make_grid_coords(
    image_shape: tuple[int, int],
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Build the full (h*w, 2) coordinate grid for evaluating a trained model
    at every pixel — used to reconstruct the image from the learned field."""
    h, w = image_shape
    ys, xs = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    return torch.stack([xs, ys], dim=-1).reshape(-1, 2)
