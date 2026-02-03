"""Hybrid analytical + ML residual workflow for the pressure sensor.

Workflow:
1) Load truth data from Excel.
2) Generate analytical pressure sweep (common P-grid).
3) Compute analytical capacitance for each (R, t3) group.
4) Build residual targets y_residual = C_truth - C_analytical.
5) Train ML model on inference-available features.
6) Use ML at inference time: C_hybrid = C_analytical + C_residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.append(str(MODULE_DIR))

from sensor_analytical import AnalyticalSensorModel, SensorParams


@dataclass(frozen=True)
class DataRanges:
    """Parameter filters for training and validation splits (SI units)."""

    radius_min_m: float = 300e-6
    radius_max_m: float = 500e-6
    t3_min_m: float = 2e-6
    t3_max_m: float = 10e-6


@dataclass(frozen=True)
class ResidualNetworkConfig:
    """Detailed MLP configuration for residual learning."""

    hidden_layer_sizes: Tuple[int, ...] = (128, 128, 64, 32)
    activation: str = "relu"
    alpha: float = 1e-4
    learning_rate: str = "adaptive"
    learning_rate_init: float = 1e-3
    max_iter: int = 2000
    random_state: int = 42


class ResidualHybridPipeline:
    """Object-oriented workflow for analytical + residual ML modeling."""

    def __init__(
        self,
        base_params: Dict[str, float],
        ranges: DataRanges | None = None,
        network_config: ResidualNetworkConfig | None = None,
    ) -> None:
        self.base_params = base_params
        self.ranges = ranges or DataRanges()
        self.network_config = network_config or ResidualNetworkConfig()
        self.model: Pipeline | None = None

    @staticmethod
    def _maybe_convert_units(values: np.ndarray, name: str) -> np.ndarray:
        """Convert units if values look like um or fF; keep SI otherwise."""

        values = np.asarray(values, dtype=float)
        if name in {"radius_m", "t3_m"} and values.max() > 1e-3:
            return values * 1e-6
        if name == "capacitance_f" and values.max() > 1e-9:
            return values * 1e-15
        return values

    def load_truth_data(
        self,
        excel_path: str,
        sheet_name: int | str = 0,
        column_map: Dict[str, str] | None = None,
    ) -> pd.DataFrame:
        """Load truth data from Excel.

        Expected columns (after mapping):
        - P (Pa)
        - C_truth (F or fF)
        - R (m or um)
        - t3 (m or um)
        """

        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        if column_map:
            df = df.rename(columns=column_map)

        required = {"P", "C_truth", "R", "t3"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}. Found: {sorted(df.columns)}"
            )

        df = df.copy()
        df["R"] = self._maybe_convert_units(df["R"].to_numpy(), "radius_m")
        df["t3"] = self._maybe_convert_units(df["t3"].to_numpy(), "t3_m")
        df["C_truth"] = self._maybe_convert_units(df["C_truth"].to_numpy(), "capacitance_f")

        return df

    def build_dataset(
        self,
        df_truth: pd.DataFrame,
        pressures_pa: Iterable[float],
        allow_interpolation: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Construct training dataset for residual learning."""

        pressures_pa = np.asarray(list(pressures_pa), dtype=float)

        X_list = []
        y_list = []

        for (radius, t3), group in df_truth.groupby(["R", "t3"]):
            if not (self.ranges.radius_min_m <= radius <= self.ranges.radius_max_m):
                continue
            if not (self.ranges.t3_min_m <= t3 <= self.ranges.t3_max_m):
                continue

            group_sorted = group.sort_values("P")
            P_truth = group_sorted["P"].to_numpy(dtype=float)
            C_truth = group_sorted["C_truth"].to_numpy(dtype=float)

            if np.array_equal(P_truth, pressures_pa):
                C_truth_aligned = C_truth
            elif allow_interpolation:
                C_truth_aligned = np.interp(pressures_pa, P_truth, C_truth)
            else:
                raise ValueError("Pressure grids do not align and interpolation is disabled.")

            params = SensorParams(
                radius_m=radius,
                t3_m=t3,
                **self.base_params,
            )
            analytical_model = AnalyticalSensorModel(params)
            C_analytical = analytical_model.capacitance(pressures_pa)

            y_residual = C_truth_aligned - C_analytical

            # Features must be available at inference time.
            X = np.column_stack(
                [
                    pressures_pa,
                    np.full_like(pressures_pa, radius),
                    np.full_like(pressures_pa, t3),
                    C_analytical,
                ]
            )

            X_list.append(X)
            y_list.append(y_residual)

        if not X_list:
            raise ValueError("No data matched the requested radius/t3 ranges.")

        return np.vstack(X_list), np.concatenate(y_list)

    @staticmethod
    def split_train_validation(
        X: np.ndarray,
        y: np.ndarray,
        validation_fraction: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Hold out a portion of the numerical data for validation."""

        return train_test_split(X, y, test_size=validation_fraction, random_state=random_state)

    def train_residual_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """Train a detailed residual MLP model."""

        cfg = self.network_config
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=cfg.hidden_layer_sizes,
                        activation=cfg.activation,
                        alpha=cfg.alpha,
                        learning_rate=cfg.learning_rate,
                        learning_rate_init=cfg.learning_rate_init,
                        max_iter=cfg.max_iter,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
        model.fit(X_train, y_train)
        self.model = model
        return model

    @staticmethod
    def evaluate_model(model: Pipeline, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """Compute validation MSE on residuals."""

        pred = model.predict(X_val)
        return mean_squared_error(y_val, pred)

    def hybrid_predict(self, pressures_pa: Iterable[float], params: SensorParams) -> np.ndarray:
        """Inference: C_hybrid = C_analytical + C_residual."""

        if self.model is None:
            raise ValueError("Residual model has not been trained.")

        pressures_pa = np.asarray(list(pressures_pa), dtype=float)
        analytical_model = AnalyticalSensorModel(params)
        C_analytical = analytical_model.capacitance(pressures_pa)
        X = np.column_stack(
            [
                pressures_pa,
                np.full_like(pressures_pa, params.radius_m),
                np.full_like(pressures_pa, params.t3_m),
                C_analytical,
            ]
        )
        C_residual = self.model.predict(X)
        return C_analytical + C_residual


def example_run(excel_path: str) -> None:
    """Example workflow wiring (train + validate)."""

    base_params = dict(
        gap_m=10e-6,
        t1_m=1e-6,
        t2_m=0.2e-6,
        t4_m=0.71e-6,
        E_p=3.2e9,
        E_au=70e9,
        v_p=0.33,
        v_au=0.44,
        e_p=3.15,
        e_air=1.0,
        e_0=8.85e-12,
    )

    pipeline = ResidualHybridPipeline(base_params=base_params)
    pressures_pa = AnalyticalSensorModel.generate_pressure_grid()
    df_truth = pipeline.load_truth_data(excel_path)

    X, y = pipeline.build_dataset(df_truth, pressures_pa)
    X_train, X_val, y_train, y_val = pipeline.split_train_validation(
        X, y, validation_fraction=0.2
    )
    model = pipeline.train_residual_model(X_train, y_train)
    mse = pipeline.evaluate_model(model, X_val, y_val)

    print(f"Validation residual MSE: {mse:.3e}")


if __name__ == "__main__":
    example_run("400um_4um.xlsx")
