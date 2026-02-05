"""Generate comparison plots and metrics for a pressure sweep."""

from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import SensorConfig
from data import load_fea_truth
from inference import HybridPredictor
from physics import AnalyticalModel, GeometryParams


def _filter_geometry(rows: list[dict[str, float]], t3_um: float, radius_um: float) -> list[dict[str, float]]:
    return [
        row for row in rows if row["t3_um"] == t3_um and row["radius_um"] == radius_um
    ]


def _rmse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def run_sweep(t3_um: float = 4.0, radius_um: float = 400.0) -> Tuple[float, float]:
    cfg = SensorConfig()
    truth_rows = load_fea_truth(cfg)
    geometry_rows = _filter_geometry(truth_rows, t3_um, radius_um)
    if not geometry_rows:
        raise ValueError(f"No FEA rows found for t3={t3_um}, radius={radius_um}.")

    geometry_rows.sort(key=lambda row: row["pressure_pa"])
    pressures = np.array([row["pressure_pa"] for row in geometry_rows], dtype=float)
    numerical = np.array([row["capacitance_fF"] for row in geometry_rows], dtype=float)

    analytical_model = AnalyticalModel(cfg)
    geom = GeometryParams(radius_m=radius_um * 1e-6, t3_parylene_m=t3_um * 1e-6)
    analytical = analytical_model.capacitance_sweep(pressures, geom) * 1e15

    predictor = HybridPredictor(cfg)
    hybrid = predictor.predict_hybrid_batch(t3_um, radius_um, pressures)

    rmse_analytical = _rmse(analytical, numerical)
    rmse_hybrid = _rmse(hybrid, numerical)

    os.makedirs("model_data", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(pressures, numerical, label="Numerical (FEA)", linewidth=2)
    plt.plot(pressures, analytical, label="Analytical", linestyle="--")
    plt.plot(pressures, hybrid, label="Hybrid", linestyle="-.")
    plt.xlabel("Pressure (Pa)")
    plt.ylabel("Capacitance (fF)")
    plt.title(f"Pressure Sweep (t3={t3_um}um, radius={radius_um}um)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"model_data/pressure_sweep_t3_{int(t3_um)}_radius_{int(radius_um)}.png", dpi=200)
    plt.close()

    print(f"RMSE Analytical vs Numerical: {rmse_analytical:.6f} fF")
    print(f"RMSE Hybrid vs Numerical: {rmse_hybrid:.6f} fF")
    return rmse_analytical, rmse_hybrid


if __name__ == "__main__":
    run_sweep()
