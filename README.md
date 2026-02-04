# MEMS Pressure Sensor Hybrid Model

This repository contains a physics-based analytical capacitance model plus a residual ML model that learns the FEM/experimental error. The workflow keeps physics modular while allowing the ML model to correct the analytical baseline.

## Repository layout

```
config.py           # Constants & hyperparameters
physics.py          # Analytical model (Python translation of MATLAB)
data.py             # Excel parser, residual dataset, split & scalers
network.py          # Residual MLP definition
train.py            # Training loop (saves best model + scalers)
inference.py        # Hybrid inference (analytical + ML residual)
model_data/         # FEA CSV data, analytical sweep, scalers, trained model
Previous Files/     # Archived MATLAB (binary docs removed from git)
```

## Typical workflow

1. **Generate analytical sweep** (pressure-capacitance curves for t3/radius grid):
   ```bash
   python physics.py
   ```
   This writes `model_data/analytical_sweep.csv`.

2. **Use FEA CSV data** (already stored in repo):
   ```bash
   cat model_data/fea_truth.csv | head
   ```

3. **Train residual model**:
   ```bash
   python train.py
   ```

4. **Hybrid inference**:
   ```bash
   python inference.py
   ```

The hybrid output is:
```
C_hybrid = C_analytical + C_residual
```

`physics.py` can be run on its own to export analytical data for downstream training or recordkeeping.
