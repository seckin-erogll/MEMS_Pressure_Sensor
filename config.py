"""Configuration for MEMS pressure sensor analytical + residual ML workflow."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SensorConfig:
    # Geometry (meters)
    radius_range_um: Tuple[float, float] = (300.0, 500.0)
    radius_step_um: float = 50.0
    gap_m: float = 10e-6
    t1_parylene_m: float = 1.0e-6
    t2_gold_m: float = 0.2e-6
    t3_parylene_range_um: Tuple[float, float] = (2.0, 10.0)
    t3_step_um: float = 2.0
    t4_bottom_parylene_m: float = 0.71e-6

    # Pressure sweep
    pressure_step_pa: float = 100.0
    pressure_max_pa: float = 14_000.0

    # Mechanical properties
    e_parylene_pa: float = 3.2e9
    e_gold_pa: float = 70e9
    poisson_parylene: float = 0.33
    poisson_gold: float = 0.44

    # Electrical properties
    eps_parylene: float = 3.15
    eps_air: float = 1.0
    eps_0: float = 8.85e-12

    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 200

    # File locations (CSV is tracked to avoid binary artifacts)
    fea_csv_path: str = "model_data/fea_truth.csv"
    fea_excel_path: str | None = None
    analytical_csv_path: str = "model_data/analytical_sweep.csv"
    scaler_path: str = "model_data/scalers.json"
    model_path: str = "model_data/best_model.pth"
