"""Analytical capacitance model translated from Sensor_Analytical_Model.m.

Workflow intent:
- Provide C_analytical(P, params) as the physics baseline.
- Export a pressure-capacitance sweep for training or recordkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import math
import numpy as np

from config import SensorConfig


@dataclass(frozen=True)
class GeometryParams:
    radius_m: float
    t3_parylene_m: float


class AnalyticalModel:
    def __init__(self, config: SensorConfig):
        self.cfg = config

    def _linear_spring_coeff(self, radius_m: float, t3_m: float) -> float:
        cfg = self.cfg
        e = np.array([cfg.e_parylene_pa, cfg.e_gold_pa, cfg.e_parylene_pa], dtype=float)
        t = np.array([cfg.t1_parylene_m, cfg.t2_gold_m, t3_m], dtype=float)
        v = np.array([cfg.poisson_parylene, cfg.poisson_gold, cfg.poisson_parylene], dtype=float)

        c_1 = (e[0] * t[0]) / (1 - v[0] ** 2)
        c_2 = (e[1] * t[1]) / (1 - v[1] ** 2)
        c_3 = (e[2] * t[2]) / (1 - v[2] ** 2)

        h_tot = t.sum()
        d = -(c_2 * t[0] + c_3 * (t[0] + t[1])) / (2 * (c_1 + c_2 + c_3))
        d = abs(d)

        b_1 = (h_tot / 2) - d
        b_2 = t[2] - b_1
        b_4 = (h_tot / 2) + d
        b_3 = b_4 - t[0]

        sum_1 = (e[2] * (b_2**3 - b_1**3)) / (1 - v[2] ** 2)
        sum_2 = (e[1] * (b_3**3 - b_2**3)) / (1 - v[1] ** 2)
        sum_3 = (e[0] * (b_4**3 - b_3**3)) / (1 - v[0] ** 2)
        k_1 = sum_1 + sum_2 + sum_3

        return (64 * math.pi * k_1) / (radius_m**2)

    def _cubic_spring_coeff(self, radius_m: float, t3_m: float) -> float:
        cfg = self.cfg
        e = np.array([cfg.e_parylene_pa, cfg.e_gold_pa, cfg.e_parylene_pa], dtype=float)
        t = np.array([cfg.t1_parylene_m, cfg.t2_gold_m, t3_m], dtype=float)
        v = np.array([cfg.poisson_parylene, cfg.poisson_gold, cfg.poisson_parylene], dtype=float)

        d_values = (e * t**3) / (12 * (1 - v**2))

        ps = np.prod(t**2)
        nom = np.sum((d_values * v * ps) / (t**2))
        den = np.sum((d_values * ps) / (t**2))
        v_m = nom / den

        cc = np.sum((d_values * ps) / (t**2))
        const = (81 * math.pi) * (-2109 * v_m**2 + 3210 * v_m + 5679) / (625 * radius_m**2)
        return const * cc / ps

    def _solve_deflection(self, pressure_pa: float, k_lin: float, k_cubic: float, radius_m: float) -> float:
        """Solve for average deflection (w_avg) using bisection."""
        def q(w_avg: float) -> float:
            return (k_lin * w_avg + k_cubic * (w_avg**3)) / (math.pi * radius_m**2)

        lo, hi = 0.0, 50e-6
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if q(mid) < pressure_pa:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    def _capacitance_for_pressure(self, pressure_pa: float, geom: GeometryParams) -> float:
        cfg = self.cfg
        k_lin = self._linear_spring_coeff(geom.radius_m, geom.t3_parylene_m)
        k_cubic = self._cubic_spring_coeff(geom.radius_m, geom.t3_parylene_m)
        w_avg = self._solve_deflection(pressure_pa, k_lin, k_cubic, geom.radius_m)
        w_0 = 3.0 * w_avg

        if w_0 <= 0:
            return 0.0

        constant1 = 1 - math.sqrt(cfg.gap_m / w_0)
        if w_0 < cfg.gap_m:
            xtm = 0.0
        else:
            xtm = geom.radius_m * math.sqrt(constant1)

        h = cfg.gap_m + ((geom.t3_parylene_m + cfg.t4_bottom_parylene_m) / cfg.eps_parylene)
        c_contact = (math.pi * xtm**2 * cfg.eps_0) / (h - cfg.gap_m)

        r = np.linspace(xtm, geom.radius_m, 2000)
        denom = h - w_0 * (1 - (r**2 / geom.radius_m**2)) ** 2
        denom = np.where(denom <= 0, np.nan, denom)
        integrand = r / denom
        c_no_contact = 2 * math.pi * cfg.eps_0 * np.trapezoid(integrand, r)

        c_total = c_contact + c_no_contact
        return float(c_total)

    def capacitance_sweep(self, pressures_pa: Iterable[float], geom: GeometryParams) -> np.ndarray:
        values = [self._capacitance_for_pressure(p, geom) for p in pressures_pa]
        return np.array(values, dtype=float)

    def generate_pressure_grid(self) -> np.ndarray:
        return np.arange(0.0, self.cfg.pressure_max_pa + self.cfg.pressure_step_pa, self.cfg.pressure_step_pa)

    def generate_geometry_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        t3_um = np.arange(self.cfg.t3_parylene_range_um[0], self.cfg.t3_parylene_range_um[1] + 0.1, self.cfg.t3_step_um)
        radius_um = np.arange(self.cfg.radius_range_um[0], self.cfg.radius_range_um[1] + 0.1, self.cfg.radius_step_um)
        return t3_um, radius_um

    def export_pressure_capacitance_sweep(self, output_csv: str) -> None:
        pressures = self.generate_pressure_grid()
        t3_um, radius_um = self.generate_geometry_grid()

        lines = ["t3_um,radius_um,pressure_pa,capacitance_fF"]
        for t3 in t3_um:
            for radius in radius_um:
                geom = GeometryParams(radius_m=radius * 1e-6, t3_parylene_m=t3 * 1e-6)
                capacitance = self.capacitance_sweep(pressures, geom) * 1e15
                for p, c in zip(pressures, capacitance):
                    lines.append(f"{t3:.3f},{radius:.1f},{p:.1f},{c:.6f}")

        with open(output_csv, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))


if __name__ == "__main__":
    cfg = SensorConfig()
    model = AnalyticalModel(cfg)
    model.export_pressure_capacitance_sweep(cfg.analytical_csv_path)
    print(f"Saved analytical sweep to {cfg.analytical_csv_path}")
