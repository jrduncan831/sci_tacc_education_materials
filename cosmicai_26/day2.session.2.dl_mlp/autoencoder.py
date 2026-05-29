"""MLP autoencoder for Galaxy10 images.

Encoder maps a flattened (69 * 69 * 3 = 14283)-dim image to a low-dim
latent vector through a stack of Linear+activation blocks. Decoder mirrors
the encoder back to image space and applies a final tanh so reconstructions
land in [-1, 1] to match normalized targets.

Configuration is exposed via AutoencoderConfig so students can swap latent
size, hidden widths, or activation without modifying this file.
"""

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class AutoencoderConfig:
    input_shape:   tuple[int, int, int] = (69, 69, 3)     # H, W, C
    hidden_widths: tuple[int, ...]      = (512, 128)      # encoder widths; decoder mirrors
    latent_dim:    int                  = 16
    activation:    str                  = "relu"          # "relu" or "leakyrelu"


def _act(kind: str) -> nn.Module:
    if kind == "relu":
        return nn.ReLU()
    if kind == "leakyrelu":
        return nn.LeakyReLU(0.2)
    raise ValueError(f"Unknown activation: {kind!r}")


def build_autoencoder(config: AutoencoderConfig) -> tuple[nn.Module, nn.Module]:
    """Build (encoder, decoder) as separate nn.Sequential modules.

    Encoder: Flatten -> Linear -> act -> ... -> Linear(latent_dim)
    Decoder: Linear -> act -> ... -> Linear(in_dim) -> Tanh -> Unflatten
    """
    in_dim = config.input_shape[0] * config.input_shape[1] * config.input_shape[2]

    # --- Encoder ---
    enc_layers: list[nn.Module] = [
        nn.Flatten(start_dim=1),
        nn.Linear(in_dim, config.hidden_widths[0]),
        _act(config.activation),
    ]
    for w_in, w_out in zip(config.hidden_widths[:-1], config.hidden_widths[1:]):
        enc_layers += [nn.Linear(w_in, w_out), _act(config.activation)]
    enc_layers += [nn.Linear(config.hidden_widths[-1], config.latent_dim)]
    encoder = nn.Sequential(*enc_layers)

    # --- Decoder (mirror) ---
    reversed_widths = config.hidden_widths[::-1]
    dec_layers: list[nn.Module] = [
        nn.Linear(config.latent_dim, reversed_widths[0]),
        _act(config.activation),
    ]
    for w_in, w_out in zip(reversed_widths[:-1], reversed_widths[1:]):
        dec_layers += [nn.Linear(w_in, w_out), _act(config.activation)]
    dec_layers += [
        nn.Linear(reversed_widths[-1], in_dim),
        nn.Tanh(),
        nn.Unflatten(dim=1, unflattened_size=config.input_shape),
    ]
    decoder = nn.Sequential(*dec_layers)

    return encoder, decoder
