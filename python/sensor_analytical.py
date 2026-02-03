"""Analytical model for the capacitive pressure sensor diaphragm.

This is a Python translation of Sensor_Analytical_Model.m focused on:
- Computing the analytical capacitance for a pressure sweep.
- Keeping the API clean for hybrid ML residual workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


@dataclass(frozen=True)
class SensorParams:
    """Material and geometry parameters (SI units)."""

    # Geometry
    radius_m: float
    gap_m: float
    t1_m: float
    t2_m: float
    t3_m: float
    t4_m: float

    # Mechanical
    E_p: float
    E_au: float
    v_p: float
    v_au: float

    # Electrical
    e_p: float
    e_air: float
    e_0: float


class AnalyticalSensorModel:
    """Object-oriented analytical model mirroring the MATLAB script."""

    def __init__(self, params: SensorParams) -> None:
        self.params = params

    def _linear_spring_coefficient(self) -> float:
        params = self.params
        E = np.array([params.E_p, params.E_au, params.E_p], dtype=float)
        v = np.array([params.v_p, params.v_au, params.v_p], dtype=float)
        t = np.array([params.t1_m, params.t2_m, params.t3_m], dtype=float)

        c1 = (E[0] * t[0]) / (1 - v[0] ** 2)
        c2 = (E[1] * t[1]) / (1 - v[1] ** 2)
        c3 = (E[2] * t[2]) / (1 - v[2] ** 2)

        h_tot = t.sum()
        d = abs(-(c2 * t[0] + c3 * (t[0] + t[1])) / (2 * (c1 + c2 + c3)))

        b1 = (h_tot / 2) - d
        b2 = t[2] - b1
        b4 = (h_tot / 2) + d
        b3 = b4 - t[0]

        sum1 = (E[2] * (b2**3 - b1**3)) / (1 - v[2] ** 2)
        sum2 = (E[1] * (b3**3 - b2**3)) / (1 - v[1] ** 2)
        sum3 = (E[0] * (b4**3 - b3**3)) / (1 - v[0] ** 2)

        k1 = sum1 + sum2 + sum3
        return (64 * np.pi * k1) / (params.radius_m**2)

    def _cubic_spring_coefficient(self) -> float:
        params = self.params
        E = np.array([params.E_p, params.E_au, params.E_p], dtype=float)
        v = np.array([params.v_p, params.v_au, params.v_p], dtype=float)
        t = np.array([params.t1_m, params.t2_m, params.t3_m], dtype=float)

        D = (E * (t**3)) / (12 * (1 - v**2))
        ps = np.prod(t**2)

        nom = sum((D[i] * v[i] * ps) / (t[i] ** 2) for i in range(3))
        den = sum((D[i] * ps) / (t[i] ** 2) for i in range(3))
        v_m = nom / den

        cc = sum(D[i] * ps / (t[i] ** 2) for i in range(3))
        const = (81 * np.pi) * (-2109 * v_m**2 + 3210 * v_m + 5679) / (
            625 * params.radius_m**2
        )
        return const * cc / ps

    def capacitance(self, pressures_pa: Iterable[float]) -> np.ndarray:
        """Compute analytical capacitance (F) for a pressure sweep."""

        params = self.params
        pressures_pa = np.asarray(list(pressures_pa), dtype=float)
        k_lin = self._linear_spring_coefficient()
        k_cubic = self._cubic_spring_coefficient()

        C = np.zeros_like(pressures_pa, dtype=float)

        for idx, pressure in enumerate(pressures_pa):
            def q(w_avg: float) -> float:
                return ((k_lin * w_avg) + (k_cubic * (w_avg**3))) / (
                    np.pi * params.radius_m**2
                )

            def f(w_avg: float) -> float:
                return q(w_avg) - pressure

            w_avg = brentq(f, 0.0, 50e-6)
            w0 = 3 * w_avg

            if w0 < params.gap_m:
                xtm = 0.0
            else:
                xtm = params.radius_m * np.sqrt(1 - np.sqrt(params.gap_m / w0))

            h = params.gap_m + ((params.t3_m + params.t4_m) / params.e_p)
            C_contact = (np.pi * xtm**2 * params.e_0) / (h - params.gap_m)

            def integrand(r: float) -> float:
                return r / (h - w0 * (1 - (r**2 / params.radius_m**2)) ** 2)

            C_noncontact = 2 * np.pi * params.e_0 * quad(
                integrand, xtm, params.radius_m
            )[0]
            C[idx] = C_contact + C_noncontact

        return C

    @staticmethod
    def generate_pressure_grid(
        p_min: float = 0.0, p_max: float = 10_000.0, dp: float = 100.0
    ) -> np.ndarray:
        """Generate the analytical pressure sweep used for alignment."""

        return np.arange(p_min, p_max + dp, dp, dtype=float)
