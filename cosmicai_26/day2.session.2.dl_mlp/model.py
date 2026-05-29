"""Configurable MLP for neural-field regression: (coords) -> values.

Two activation choices, selected via the config:

  - "relu": standard MLP with default PyTorch init.
  - "sin":  SIREN-style (Sitzmann et al. 2020) with sin activations and the
            paper's specific weight init. The first sin layer applies
            sin(omega_0 * x); subsequent layers apply plain sin(x). The init
            is essential — without it, sin-activated networks vanish/explode
            within a few training steps.

The final layer is always linear (no output activation) so the model can
predict any real-valued target.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass
class MLPConfig:
    in_dim:        int   = 2
    out_dim:       int   = 1
    hidden_width:  int   = 256
    hidden_layers: int   = 4         # number of Linear+activation blocks (output Linear is added on top)
    activation:    str   = "relu"    # "relu" or "sin"
    first_omega:   float = 30.0      # first-layer frequency scale; used only when activation="sin"


class Sine(nn.Module):
    """sin(omega * x) activation."""

    def __init__(self, omega: float = 1.0):
        super().__init__()
        self.omega = omega

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega * x)


def build_mlp(config: MLPConfig) -> nn.Module:
    """Build an MLP from a config. Returns an nn.Sequential with a linear
    output layer. If config.activation == 'sin', SIREN weight init is
    applied automatically — see _siren_init."""
    if config.activation not in ("relu", "sin"):
        raise ValueError(f"Unknown activation: {config.activation!r}")

    def act(first: bool = False) -> nn.Module:
        if config.activation == "relu":
            return nn.ReLU()
        return Sine(omega=config.first_omega if first else 1.0)

    layers: list[nn.Module] = [nn.Linear(config.in_dim, config.hidden_width), act(first=True)]
    for _ in range(config.hidden_layers - 1):
        layers += [nn.Linear(config.hidden_width, config.hidden_width), act()]
    layers += [nn.Linear(config.hidden_width, config.out_dim)]
    model = nn.Sequential(*layers)

    if config.activation == "sin":
        _siren_init(model, config.first_omega)
    return model


def _siren_init(model: nn.Module, first_omega: float) -> None:
    """Sitzmann et al. (2020) initialization for sin-activated MLPs:
        first Linear:      w ~ U(-1/d_in, 1/d_in)
        subsequent Linear: w ~ U(-sqrt(6/d_in)/omega_0, +sqrt(6/d_in)/omega_0)
    Biases zeroed throughout."""
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for i, layer in enumerate(linears):
        d_in = layer.in_features
        bound = 1.0 / d_in if i == 0 else math.sqrt(6.0 / d_in) / first_omega
        with torch.no_grad():
            layer.weight.uniform_(-bound, bound)
            layer.bias.zero_()
