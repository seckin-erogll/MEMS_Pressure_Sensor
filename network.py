"""Residual network definition.

The network predicts C_residual so the hybrid output is:
C_hybrid = C_analytical + C_residual
"""

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise ImportError("PyTorch is required for network.py") from exc


class ResidualCapacitanceNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.model(features)
