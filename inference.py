"""Hybrid inference pipeline.

C_hybrid(P) = C_analytical(P) + C_residual(P)
"""

from __future__ import annotations

import numpy as np

from config import SensorConfig
from data import load_scalers
from network import ResidualCapacitanceNet
from physics import AnalyticalModel, GeometryParams

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for inference") from exc


def _scale_features(features: np.ndarray, x_scaler: dict) -> np.ndarray:
    mean = np.array(x_scaler["mean"], dtype=float)
    std = np.array(x_scaler["std"], dtype=float)
    return (features - mean) / std


def _unscale_targets(values: np.ndarray, y_scaler: dict) -> np.ndarray:
    mean = y_scaler["mean"][0]
    std = y_scaler["std"][0]
    return values * std + mean


def hybrid_capacitance(pressure_pa: float, radius_um: float, t3_um: float) -> float:
    cfg = SensorConfig()

    # a) Analytical baseline
    geom = GeometryParams(radius_m=radius_um * 1e-6, t3_parylene_m=t3_um * 1e-6)
    analytical = AnalyticalModel(cfg)
    c_analytical = analytical.capacitance_sweep([pressure_pa], geom)[0] * 1e15

    # b) Load ML residual
    x_scaler, y_scaler = load_scalers(cfg.scaler_path)
    features = np.array([[t3_um, radius_um, pressure_pa]], dtype=float)
    features_scaled = _scale_features(features, x_scaler)

    model = ResidualCapacitanceNet()
    model.load_state_dict(torch.load(cfg.model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        residual_scaled = model(torch.tensor(features_scaled, dtype=torch.float32)).numpy().flatten()

    c_residual = _unscale_targets(residual_scaled, y_scaler)[0]
    return float(c_analytical + c_residual)


if __name__ == "__main__":
    example = hybrid_capacitance(pressure_pa=1000.0, radius_um=300.0, t3_um=4.0)
    print(f"Hybrid capacitance (fF): {example:.3f}")
