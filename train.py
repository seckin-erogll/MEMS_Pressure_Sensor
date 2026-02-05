"""Training workflow for the residual model.

Step-by-step (workflow-driven) outline:
1) Load FEA truth data (P, C_truth) for each (t3, radius) sweep.
2) Compute C_analytical(P) on the same P-grid.
3) Build residual targets: y_residual = C_truth - C_analytical.
4) Split by full sweeps to avoid leakage (20% for validation).
5) Fit scalers on training data only; apply to val/test.
6) Train ML to map x -> y_residual; save best model + scalers.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Tuple

import numpy as np

from config import SensorConfig
from data import (
    ResidualSample,
    build_residual_dataset,
    fit_scalers,
    load_fea_truth,
    save_scalers,
    split_by_geometry,
)
from network import ResidualCapacitanceNet
from physics import AnalyticalModel, GeometryParams

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required to train the residual model") from exc


def _build_analytical_lookup(cfg: SensorConfig) -> Dict[Tuple[float, float, float], float]:
    model = AnalyticalModel(cfg)
    pressures = model.generate_pressure_grid()
    t3_um, radius_um = model.generate_geometry_grid()

    lookup: Dict[Tuple[float, float, float], float] = {}
    for t3 in t3_um:
        for radius in radius_um:
            geom = GeometryParams(radius_m=radius * 1e-6, t3_parylene_m=t3 * 1e-6)
            cap = model.capacitance_sweep(pressures, geom) * 1e15
            for p, c in zip(pressures, cap):
                lookup[(float(t3), float(radius), float(p))] = float(c)
    return lookup


def _scale_features(samples: list[ResidualSample], x_scaler: Dict[str, list[float]], y_scaler: Dict[str, list[float]]):
    x_mean = np.array(x_scaler["mean"], dtype=float)
    x_std = np.array(x_scaler["std"], dtype=float)
    y_mean = y_scaler["mean"][0]
    y_std = y_scaler["std"][0]

    x = np.array([[s.t3_um, s.radius_um, s.pressure_pa] for s in samples], dtype=float)
    y = np.array([s.residual_fF for s in samples], dtype=float)

    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - y_mean) / y_std
    return x_scaled, y_scaled


def train() -> None:
    cfg = SensorConfig()

    # 1) Load truth data
    truth_rows = load_fea_truth(cfg)

    # 2) Analytical baseline on the same grid
    analytical_lookup = _build_analytical_lookup(cfg)

    # 3) Residual target construction
    residual_samples = build_residual_dataset(truth_rows, analytical_lookup)

    # 4) Sweep-aware split (20% of geometry sweeps for validation)
    train_samples, val_samples = split_by_geometry(residual_samples, validation_ratio=0.2)

    # 5) Normalization
    x_scaler, y_scaler = fit_scalers(train_samples)
    save_scalers(x_scaler, y_scaler, cfg.scaler_path)

    x_train, y_train = _scale_features(train_samples, x_scaler, y_scaler)
    x_val, y_val = _scale_features(val_samples, x_scaler, y_scaler)

    # 6) Training loop (minimal)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualCapacitanceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    x_train_t = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    x_val_t = torch.tensor(x_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    best_loss = math.inf
    for epoch in range(cfg.epochs):
        start_time = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        pred = model(x_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        epoch_time = time.perf_counter() - start_time
        print(
            "Epoch "
            f"{epoch + 1}/{cfg.epochs} "
            f"- train loss: {train_loss:.6f} "
            f"- val loss: {val_loss:.6f} "
            f"- time: {epoch_time:.2f}s"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), cfg.model_path)

    print(f"Training complete. Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    train()
