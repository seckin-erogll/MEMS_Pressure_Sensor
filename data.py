"""Data loading, Excel parsing, and split logic for residual training."""

from __future__ import annotations

import csv
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from config import SensorConfig


@dataclass(frozen=True)
class ResidualSample:
    t3_um: float
    radius_um: float
    pressure_pa: float
    residual_fF: float


def _load_shared_strings(zip_handle: zipfile.ZipFile) -> List[str]:
    ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    if "xl/sharedStrings.xml" not in zip_handle.namelist():
        return []
    sst = ET.fromstring(zip_handle.read("xl/sharedStrings.xml"))
    strings = []
    for si in sst.findall("ns:si", ns):
        t = si.find(".//ns:t", ns)
        strings.append(t.text if t is not None else "")
    return strings


def _cell_value(cell: ET.Element, shared: List[str], ns: Dict[str, str]) -> str | None:
    v = cell.find("ns:v", ns)
    if v is None:
        return None
    if cell.attrib.get("t") == "s":
        return shared[int(v.text)]
    return v.text


def _parse_sheet_to_grid(sheet_xml: bytes, shared: List[str]) -> Dict[str, str | None]:
    ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    data = ET.fromstring(sheet_xml)
    cell_map: Dict[str, str | None] = {}
    for row in data.findall("ns:sheetData/ns:row", ns):
        for cell in row.findall("ns:c", ns):
            cell_map[cell.attrib.get("r")] = _cell_value(cell, shared, ns)
    return cell_map


def extract_fea_truth_to_csv(cfg: SensorConfig) -> None:
    """Parse the Excel workbook into a normalized CSV table.

    Output columns: t3_um, radius_um, pressure_pa, capacitance_fF
    """
    if not cfg.fea_excel_path:
        raise FileNotFoundError("No Excel path configured for FEA extraction.")

    with zipfile.ZipFile(cfg.fea_excel_path) as zf:
        shared = _load_shared_strings(zf)
        sheet_xml = zf.read("xl/worksheets/sheet1.xml")

    cell_map = _parse_sheet_to_grid(sheet_xml, shared)

    block_rows = []
    for cell, value in cell_map.items():
        if isinstance(value, str) and "t_parylene2" in value:
            row = int(re.sub("[^0-9]", "", cell))
            block_rows.append((row, value))

    block_rows.sort()

    rows = []
    for block_row, label in block_rows:
        t3_match = re.search(r"t_parylene2=\s*(\d+)", label)
        if not t3_match:
            continue
        t3_um = float(t3_match.group(1))
        header_row = block_row + 1
        radii = []
        for col in ["B", "C", "D", "E", "F"]:
            value = cell_map.get(f"{col}{header_row}")
            radii.append(float(value))

        pressure_row = header_row + 1
        while True:
            pressure_value = cell_map.get(f"A{pressure_row}")
            if pressure_value is None:
                break
            if isinstance(pressure_value, str) and "t_parylene2" in pressure_value:
                break
            pressure_pa = float(pressure_value)
            for idx, col in enumerate(["B", "C", "D", "E", "F"]):
                cap_value = cell_map.get(f"{col}{pressure_row}")
                if cap_value is None:
                    continue
                rows.append(
                    {
                        "t3_um": t3_um,
                        "radius_um": radii[idx],
                        "pressure_pa": pressure_pa,
                        "capacitance_fF": float(cap_value),
                    }
                )
            pressure_row += 1

    with open(cfg.fea_csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t3_um", "radius_um", "pressure_pa", "capacitance_fF"])
        writer.writeheader()
        writer.writerows(rows)


def load_fea_truth(cfg: SensorConfig) -> List[Dict[str, float]]:
    if not _csv_exists(cfg.fea_csv_path):
        if cfg.fea_excel_path:
            extract_fea_truth_to_csv(cfg)
        else:
            raise FileNotFoundError(
                f\"Missing {cfg.fea_csv_path}. Provide CSV or set fea_excel_path to parse Excel.\"
            )
    with open(cfg.fea_csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {
                "t3_um": float(row["t3_um"]),
                "radius_um": float(row["radius_um"]),
                "pressure_pa": float(row["pressure_pa"]),
                "capacitance_fF": float(row["capacitance_fF"]),
            }
            for row in reader
        ]


def build_residual_dataset(truth_rows: Iterable[Dict[str, float]], analytical_lookup: Dict[Tuple[float, float, float], float]) -> List[ResidualSample]:
    samples = []
    for row in truth_rows:
        key = (row["t3_um"], row["radius_um"], row["pressure_pa"])
        c_analytical = analytical_lookup[key]
        residual = row["capacitance_fF"] - c_analytical
        samples.append(
            ResidualSample(
                t3_um=row["t3_um"],
                radius_um=row["radius_um"],
                pressure_pa=row["pressure_pa"],
                residual_fF=residual,
            )
        )
    return samples


def split_by_geometry(samples: List[ResidualSample], validation_ratio: float = 0.2) -> Tuple[List[ResidualSample], List[ResidualSample]]:
    """Split by full (t3, radius) sweeps to avoid leakage."""
    geometry_keys = sorted({(s.t3_um, s.radius_um) for s in samples})
    split_index = int(len(geometry_keys) * (1 - validation_ratio))
    train_keys = set(geometry_keys[:split_index])

    train, val = [], []
    for sample in samples:
        if (sample.t3_um, sample.radius_um) in train_keys:
            train.append(sample)
        else:
            val.append(sample)
    return train, val


def fit_scalers(train_samples: List[ResidualSample]) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    features = np.array([[s.t3_um, s.radius_um, s.pressure_pa] for s in train_samples], dtype=float)
    targets = np.array([s.residual_fF for s in train_samples], dtype=float)

    x_mean = features.mean(axis=0)
    x_std = features.std(axis=0) + 1e-9
    y_mean = targets.mean()
    y_std = targets.std() + 1e-9

    x_scaler = {"mean": x_mean.tolist(), "std": x_std.tolist()}
    y_scaler = {"mean": [float(y_mean)], "std": [float(y_std)]}
    return x_scaler, y_scaler


def save_scalers(x_scaler: Dict[str, List[float]], y_scaler: Dict[str, List[float]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"x": x_scaler, "y": y_scaler}, handle, indent=2)


def load_scalers(path: str) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["x"], data["y"]


def _csv_exists(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8"):
            return True
    except FileNotFoundError:
        return False
