"""Galaxy10 SDSS loader: 21,785 RGB galaxy images at 69x69 with 10 class labels.

Loads images + labels as on-device tensors with a deterministic train/val
split. Images are normalized from uint8 [0, 255] to float32 [-1, 1] so a
tanh-output autoencoder can target them directly.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

DEFAULT_IMAGES = "/scratch/projects/tacc/sci-training/cosmicai_data/galaxy10/images.npy"
DEFAULT_LABELS = "/scratch/projects/tacc/sci-training/cosmicai_data/galaxy10/labels.npy"

# Galaxy10 SDSS morphology classes (astroNN convention).
CLASS_NAMES = (
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
)


class Galaxy10(NamedTuple):
    train_images: torch.Tensor   # (N_train, 69, 69, 3) float32 in [-1, 1]
    train_labels: torch.Tensor   # (N_train,) long
    val_images:   torch.Tensor   # (N_val,   69, 69, 3)
    val_labels:   torch.Tensor   # (N_val,)


def load_galaxy10(
    images_path: str | Path = DEFAULT_IMAGES,
    labels_path: str | Path = DEFAULT_LABELS,
    device: str | torch.device = "cuda",
    train_fraction: float = 0.9,
    seed: int = 0,
) -> Galaxy10:
    """Load Galaxy10 as on-device tensors with a deterministic train/val split.

    Images: uint8 [0, 255] -> float32 [-1, 1]. Labels: long.
    All four tensors land on `device`. The whole dataset fits easily in
    GPU memory (~1.2 GB float32), so we skip DataLoader entirely.
    """
    images = np.load(images_path)                                  # (N, 69, 69, 3) uint8
    labels = np.load(labels_path)                                  # (N,) uint8

    images_t = torch.from_numpy(images).float() / 127.5 - 1.0      # -> [-1, 1]
    labels_t = torch.from_numpy(labels).long()

    n = images_t.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train = int(train_fraction * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    return Galaxy10(
        train_images=images_t[train_idx].to(device),
        train_labels=labels_t[train_idx].to(device),
        val_images=images_t[val_idx].to(device),
        val_labels=labels_t[val_idx].to(device),
    )
