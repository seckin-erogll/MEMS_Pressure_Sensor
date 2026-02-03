# MEMS_Pressure_Sensor
Robust, MEMS-based, machine learning integrated pressure sensor model for low-pressure sensing applications. 

## Python hybrid analytical + residual model

Python modules live in `python/`:

- `sensor_analytical.py`: analytical capacitance model translated from the MATLAB script.
- `hybrid_residual_pipeline.py`: residual-learning workflow (load truth data, build residuals, train, validate, infer).
- `requirements.txt`: dependencies for the Python workflow.

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
python python/hybrid_residual_pipeline.py
```

### Data expectations

The Excel file should contain the following columns (or provide a `column_map` in the loader):

- `P` (Pa)
- `C_truth` (F or fF)
- `R` (m or um)
- `t3` (m or um)

The script filters the training range to **t3 = 2–10 µm** and **R = 300–500 µm**,
then holds out a fraction of the numerical data for validation.
