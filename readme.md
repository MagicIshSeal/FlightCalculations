# Flight Performance Analysis Tool

A Python-based tool for analyzing aircraft flight performance using XFoil data and aerodynamic calculations.

## Features

- Aircraft performance analysis including:
  - Lift and drag coefficients (2D and 3D)
  - Power curves (available, required, and climb)
  - Turn performance
  - Stall speeds
  - Minimum descent rates

## Project Structure

```
FlightCalculations/
├── flight_profile_pandas.py     # Main analysis script
├── XFoil_Runner/               # XFoil automation tools
│   └── Xfoil.py               # XFoil interface
├── processed_csv/             # Output data
│   ├── aerodynamics/         # Aerodynamic analysis results
│   └── power/                # Power curve data
└── dfs_dictionary.pkl        # Cached XFoil results
```

## Requirements

- Python 3.6 - 3.10
- NumPy
- Pandas
- Matplotlib
- XFoil executable in XFoil_Runner directory

## Constants Used

- Wing span (b): 3.5 m
- Weight (W): 25 * 9.81 N
- Wing area (S): 1.45 m²
- Air density (Rho): 1.225 kg/m³
- Thrust (T): 160 N
- Base drag coefficient (Cd0): 0.175
- Oswald efficiency (e): 0.85
- Flight path angle (gamma): 16°

## Key Calculations

The script calculates:
- Reynolds numbers
- Stall velocities
- Lift coefficients (CL)
- Drag coefficients (CD)
- Power required/available
- Turn performance metrics
- Minimum descent rates

## Usage

1. Ensure all dependencies are installed:
```bash
pip install numpy pandas matplotlib
```

2. Run the script:
```bash
python flight_profile_pandas.py
```

3. When prompted, choose to:
   - Run new XFoil calculations (y)
   - Use existing cached data (n)

## Output

### Console Output
- CL maximums
- Stall speeds
- Maximum lift-to-drag ratios
- Minimum descent rates
- Turn performance analysis

### Visualizations
- Lift and drag coefficient plots
- Power curves
- Turn performance characteristics
- Drag polars

### Data Export
Results are saved in CSV format under:
- `processed_csv/aerodynamics/`
- `processed_csv/power/`

## Functions

Key aerodynamic functions:

```python
def CD(Cd0, Cl, AR, e):
    """Calculate drag coefficient"""
    return Cd0 + (Cl**2) / (np.pi * AR * e)

def rdmin(W, S, Rho, gamma, CD2CL3):
    """Calculate minimum descent rate"""
    gamma = np.radians(gamma)
    return np.sqrt((W / S) * (2 / Rho) * (CD2CL3) * (1))

def pa(V):
    """Calculate available power"""
    return T * V

def pr(CD, V):
    """Calculate required power"""
    return (CD * (1 / 2) * Rho * S * (V**2)) * V
```