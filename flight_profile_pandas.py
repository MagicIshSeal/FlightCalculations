import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "pandas_out"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Constants
b = 3.5  # m
W = 25 * 9.81  # N
S = 1.45
AR = (b**2) / S
Lambda = 0  # wing sweep angle
Rho = 1.225  # kg/m^3
T = 160  # N
Cd0 = 0.175
e = 0.85


# Load airfoil data
airfoil_data = np.genfromtxt(
    "csv_plots/NACA_Re_1239338.csv",
    delimiter=",",
    skip_header=2,
    usecols=(0, 1, 2, 3, 4, 5, 6),
)

# Create initial DataFrame
df = pd.DataFrame(
    {
        "Alpha": airfoil_data[:, 0],
        "CL_2D": airfoil_data[:, 1],
    }
)


# Helper functions for calculations
def CLa_Prandtl(Cla, AR):
    return Cla / (1 + ((Cla) / (np.pi * e * AR)))


def CLa_Kuchemann(Cla, Lambda, AR):
    Lambda = np.radians(Lambda)
    num = Cla * np.cos(Lambda)
    dnum = np.sqrt(1 + ((Cla * np.cos(Lambda)) / (np.pi * AR)) ** 2) + (
        (Cla * np.cos(Lambda)) / (np.pi * AR)
    )
    return num / dnum


# Add calculated columns
df["CL_Prandtl"] = CLa_Prandtl(df["CL_2D"], AR)
df["CL_Kuchemann"] = CLa_Kuchemann(df["CL_2D"], Lambda, AR)
df["CD"] = Cd0 + ((df["CL_2D"] ** 2) / (np.pi * AR * e))

# Calculate maximum values
max_values = pd.DataFrame(
    {
        "Method": ["2D", "Prandtl", "Kuchemann"],
        "CL_max": [df["CL_2D"].max(), df["CL_Prandtl"].max(), df["CL_Kuchemann"].max()],
        "Alpha_at_max": [
            df.loc[df["CL_2D"].idxmax(), "Alpha"],
            df.loc[df["CL_Prandtl"].idxmax(), "Alpha"],
            df.loc[df["CL_Kuchemann"].idxmax(), "Alpha"],
        ],
    }
)

# Add stall speeds
max_values["Vstall"] = np.sqrt((2 * W) / (Rho * S * max_values["CL_max"]))

# Save to CSV files
df.to_csv(os.path.join(output_dir, "lift_coefficients.csv"), index=False)
max_values.to_csv(os.path.join(output_dir, "maximum_values.csv"), index=False)

print("\nLift Coefficients Data (first 5 rows):")
print(df.head())
print("\nMaximum Values and Conditions:")
print(max_values)
