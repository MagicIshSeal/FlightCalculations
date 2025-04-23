import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from XFoil_Runner.Xfoil import run_for_Re
import os

# Constants
b = 3.5  # m
W = 25 * 9.81  # N
S = 1.45
AR = (b**2) / S
Lambda = 0  # wing sweep angle
Rho = 1.225  # kg/m^3
MAC = 0.431034483
Mu_Air = 1.7894e-5  # kg/(m*s)
T = 160  # N
Cd0 = 0.175
e = 0.85
gamma = 16

V = np.array([16, (105 / 3.6), 42])


def Reynolds_number(V, Mu_Air, MAC, Rho):
    return (Rho * V * MAC) / Mu_Air


Re = Reynolds_number(V, Mu_Air, MAC, Rho)
Re = np.array([f"{x/1e6:.3f}e6" for x in Re])

dfs = {}  # wrapper for dataframes

choice = input("Do you want to run XFoil? (y/n): ").strip().lower()

if choice == "y":
    for v, re in zip(V, Re):
        dfs[f"df_V_{v}_Re_{re}"] = run_for_Re(
            "NACA2415", re, -18.750, 19.250, 0.25, 1000
        )
    with open("dfs_dictionary.pkl", "wb") as file:
        pd.to_pickle(dfs, file)

elif choice == "n":
    with open("dfs_dictionary.pkl", "rb") as file:
        dfs = pd.read_pickle(file)


def Vstall(W, Rho, S, Cl):
    return np.sqrt((2 * W) / (Rho * S * Cl))


def CL(e, AR, cl, a0):
    k = 1 / (1 + (57.3 * (a0 / (e * np.pi * AR))))
    return cl * k


def CD(Cd0, Cl, AR, e):
    return Cd0 + (Cl**2) / (np.pi * AR * e)


def rdmin(W, S, Rho, gamma, CD2CL3):
    gamma = np.radians(gamma)
    return np.sqrt((W / S) * (2 / Rho) * (CD2CL3) * (np.cos(gamma)) ** 3)


def pa(V):
    return T * V


def pr(CD, V):
    return (CD * (1 / 2) * Rho * S * (V**2)) * V


def pc(pa, pr):
    return pa - pr


Vdmin = np.sqrt((W / S) * (2 / Rho) * (1 / np.sqrt(Cd0 * e * AR * np.pi)))
Dmin = 2 * W * np.sqrt((Cd0) / (np.pi * AR * e))

Prmin = (
    (4 / 3)
    * W
    * np.sqrt((W * S) * (2 / Rho) * np.sqrt((3 * Cd0) / ((np.pi * AR * e) ** 3)))
)
Vprmin = 1  # need to be calculated
Prbank = 1  # needs to be derived
Tturn = 1  # needs to be derived

power_curves = {}

for df_name, df in dfs.items():
    name = float(df_name.split("_")[2])  # Extract velocity from df name
    power_df = pd.DataFrame()
    alpha_zero_idk = (df["alpha"]).abs().idxmin()
    power_df["V"] = np.arange(10, 42.1, 0.1)
    power_df["Pa"] = pa(power_df["V"])
    power_df["Pr"] = pr(df["CD"][alpha_zero_idk], power_df["V"])
    power_df["Pc"] = pc(power_df["Pa"], power_df["Pr"])
    power_curves[f"power_V_{name}_Re_{df_name.split('_Re_')[1]}"] = power_df


for df_name, df in dfs.items():
    # Calculate dCL/dα using central differences
    alpha_zero_idk = (df["alpha"]).abs().idxmin()
    df["dCL_da"] = np.gradient(df["CL"], df["alpha"])
    df["CL_3D"] = CL(e, AR, df["CL"], df["dCL_da"][alpha_zero_idk])
    df["CD"] = CD(Cd0, df["CL_3D"], AR, e)
    df["CL/CD"] = df["CL_3D"] / df["CD"]
    df["CD2/CL3"] = (df["CD"] ** 2) / (df["CL_3D"] ** 3)
    df["CL3/CD2"] = (df["CL_3D"] ** 3) / (df["CD"] ** 2)

print("\nCL Maximums:")
print("-" * 50)
for i, (df_name, df) in enumerate(dfs.items()):
    cl_max = df["CL_3D"].max()
    cl_max_index = df["CL_3D"].idxmax()
    print(
        f"V = {V[i]:>6.1f} m/s: CL = {cl_max:>6.2f} at α = {df['alpha'][cl_max_index]:>6.2f}°"
    )

print("\nStall Speeds:")
print("-" * 50)
for i, (df_name, df) in enumerate(dfs.items()):
    V_stall = Vstall(W, Rho, S, df["CL_3D"].max())
    print(f"V = {V[i]:>6.1f} m/s: Vstall = {V_stall:>6.2f} m/s")

# Then print all CL/CD maximums
print("\nMaximum Lift-to-Drag Ratios:")
print("-" * 50)
for i, (df_name, df) in enumerate(dfs.items()):
    max_clcd = df["CL/CD"].max()
    max_clcd_index = df["CL/CD"].idxmax()
    print(
        f"V = {V[i]:>6.1f} m/s: CL/CD = {max_clcd:>6.2f} at α = {df['alpha'][max_clcd_index]:>6.2f}°"
    )

# Finally print all minimum descent rates
print("\nMinimum Rate of Descent:")
print("-" * 50)
for i, (df_name, df) in enumerate(dfs.items()):
    mincd2cl3 = df["CD2/CL3"].abs().min()
    mincd2cl3_index = df["CD2/CL3"].abs().idxmin()
    rate_min = rdmin(W, S, Rho, gamma, mincd2cl3)
    print(
        f"V = {V[i]:>6.1f} m/s: RD = {rate_min:>6.2f} m/s at α = {df['alpha'][mincd2cl3_index]:>6.2f}°"
    )

print("\nMisc Calculations:")
print("-" * 50)
print(f"Vdmin = {Vdmin:.2f} m/s")
print(f"Dmin = {Dmin:.2f} N")
print(f"Prmin = {Prmin:.2f} W")
print(f"Vprmin = {Vprmin:.2f} m/s")
print(f"Prbank = {Prbank:.2f} W")
print(f"Tturn = {Tturn:.2f} W")

output_dir = "processed_csv"
aero_dir = os.path.join(output_dir, "aerodynamics")
power_dir = os.path.join(output_dir, "power")
os.makedirs(aero_dir, exist_ok=True)
os.makedirs(power_dir, exist_ok=True)

# Export aerodynamic data
for df_name, df in dfs.items():
    velocity = float(df_name.split("_")[2])
    reynolds = df_name.split("_Re_")[1]
    filename = f"aero_V_{velocity:.1f}_Re_{reynolds}.csv"
    filepath = os.path.join(aero_dir, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved aerodynamic data to {filepath}")

# Export power curves
for curve_name, power_df in power_curves.items():
    velocity = float(curve_name.split("_")[2])
    reynolds = curve_name.split("_Re_")[1]
    filename = f"power_V_{velocity:.1f}_Re_{reynolds}.csv"
    filepath = os.path.join(power_dir, filename)
    power_df.to_csv(filepath, index=False)
    print(f"Saved power data to {filepath}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot CL (2D and 3D) vs alpha
for df_name, df in dfs.items():
    velocity = float(df_name.split("_")[2])  # Extract velocity from df name
    ax1.plot(df["alpha"], df["CL"], "--", label=f"$C_L$ 2D, V = {velocity:.1f} m/s")
    ax1.plot(df["alpha"], df["CL_3D"], "-", label=f"$C_L$ 3D, V = {velocity:.1f} m/s")

ax1.set_xlabel("Angle of Attack (degrees)")
ax1.set_ylabel("$C_L$")
ax1.set_title("Lift Coefficient vs Angle of Attack")
ax1.grid(True)
ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot CD vs alpha
for df_name, df in dfs.items():
    velocity = float(df_name.split("_")[2])
    ax2.plot(df["alpha"], df["CD"], label=f"V = {velocity:.1f} m/s")

ax2.set_xlabel("Angle of Attack (degrees)")
ax2.set_ylabel("$C_D$")
ax2.set_title("Drag Coefficient vs Angle of Attack")
ax2.grid(True)
ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Create power plots after your existing CL and CD plots
fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(20, 6))

# Plot Available Power vs Velocity
for df_name, power_df in power_curves.items():
    velocity = float(df_name.split("_")[2])  # Extract velocity from df name
    ax3.plot(power_df["V"], power_df["Pa"], label=f"V = {velocity:.1f} m/s")

ax3.set_xlabel("Velocity (m/s)")
ax3.set_ylabel("Available Power (W)")
ax3.set_title("Available Power vs Velocity")
ax3.grid(True)
ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot Required Power vs Velocity
for df_name, power_df in power_curves.items():
    velocity = float(df_name.split("_")[2])
    ax4.plot(power_df["V"], power_df["Pr"], label=f"V = {velocity:.1f} m/s")

ax4.set_xlabel("Velocity (m/s)")
ax4.set_ylabel("Required Power (W)")
ax4.set_title("Required Power vs Velocity")
ax4.grid(True)
ax4.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot Climb Power vs Velocity
for df_name, power_df in power_curves.items():
    velocity = float(df_name.split("_")[2])
    ax5.plot(power_df["V"], power_df["Pc"], label=f"V = {velocity:.1f} m/s")

ax5.set_xlabel("Velocity (m/s)")
ax5.set_ylabel("Climb Power (W)")
ax5.set_title("Climb Power vs Velocity")
ax5.grid(True)
ax5.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

# Add after your existing plots but before plt.show()
fig3, ((ax6, ax7), (ax8, ax9)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot CL/CD vs Alpha
for df_name, df in dfs.items():
    velocity = float(df_name.split("_")[2])
    ax6.plot(df["CD"], df["CL_3D"], label=f"V = {velocity:.1f} m/s")
ax6.set_xlabel("$C_D$")
ax6.set_ylabel("$C_L$")
ax6.set_title("Lift Coefficient vs Drag Coefficient")
ax6.grid(True)
ax6.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot CD2/CL3 vs Alpha
for df_name, df in dfs.items():
    velocity = float(df_name.split("_")[2])
    ax7.plot(df["alpha"], df["CL3/CD2"], label=f"V = {velocity:.1f} m/s")
ax7.set_xlabel("Angle of Attack (degrees)")
ax7.set_ylabel("$C_L^3/C_D^2$")
ax7.set_title("$C_L^3/C_D^2$ vs Angle of Attack")
ax7.grid(True)
ax7.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot CL/CD vs CL
for df_name, df in dfs.items():
    velocity = float(df_name.split("_")[2])
    ax8.plot(df["CL_3D"], df["CL/CD"], label=f"V = {velocity:.1f} m/s")
ax8.set_xlabel("$C_L$")
ax8.set_ylabel("$C_L/C_D$")
ax8.set_title("Lift-to-Drag Ratio vs Lift Coefficient")
ax8.grid(True)
ax8.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

# Plot CD2/CL3 vs CL
for df_name, df in dfs.items():
    velocity = float(df_name.split("_")[2])
    ax9.plot(df["CL_3D"], df["CL3/CD2"], label=f"V = {velocity:.1f} m/s")
ax9.set_xlabel("$C_L$")
ax9.set_ylabel("$C_L^3/C_D^3$")
ax9.set_title("$C_L^3/C_D^2$ vs Lift Coefficient")
ax9.grid(True)
ax9.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.show()
