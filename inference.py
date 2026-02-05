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


class HybridPredictor:
    def __init__(self, cfg: SensorConfig) -> None:
        self.cfg = cfg
        self.x_scaler, self.y_scaler = load_scalers(cfg.scaler_path)
        self.model = ResidualCapacitanceNet()
        self.model.load_state_dict(torch.load(cfg.model_path, map_location="cpu"))
        self.model.eval()
        self.analytical = AnalyticalModel(cfg)

    def predict_residual_batch(self, t3_um: float, radius_um: float, pressures_pa: np.ndarray) -> np.ndarray:
        pressures = np.asarray(pressures_pa, dtype=float)
        features = np.column_stack(
            [
                np.full_like(pressures, fill_value=t3_um, dtype=float),
                np.full_like(pressures, fill_value=radius_um, dtype=float),
                pressures,
            ]
        )
        features_scaled = _scale_features(features, self.x_scaler)
        with torch.no_grad():
            residual_scaled = (
                self.model(torch.tensor(features_scaled, dtype=torch.float32)).numpy().flatten()
            )
        return _unscale_targets(residual_scaled, self.y_scaler)

    def predict_hybrid_batch(self, t3_um: float, radius_um: float, pressures_pa: np.ndarray) -> np.ndarray:
        geom = GeometryParams(radius_m=radius_um * 1e-6, t3_parylene_m=t3_um * 1e-6)
        analytical = self.analytical.capacitance_sweep(pressures_pa, geom) * 1e15
        residual = self.predict_residual_batch(t3_um, radius_um, pressures_pa)
        return analytical + residual


def hybrid_capacitance(pressure_pa: float, radius_um: float, t3_um: float) -> float:
    cfg = SensorConfig()
    predictor = HybridPredictor(cfg)
    hybrid = predictor.predict_hybrid_batch(t3_um, radius_um, np.array([pressure_pa], dtype=float))
    return float(hybrid[0])


if __name__ == "__main__":
    example = hybrid_capacitance(pressure_pa=1000.0, radius_um=300.0, t3_um=4.0)
    print(f"Hybrid capacitance (fF): {example:.3f}")
