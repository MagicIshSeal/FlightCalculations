import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
airfoil_data = np.genfromtxt(
    "xf-n2415-il-1000000.csv",
    delimiter=",",
    skip_header=12,
    usecols=(0, 1, 2, 3, 4, 5, 6),
)
#airfoil data from airfoiltools
"""

airfoil_data = np.genfromtxt(
    "csv_plots/NACA_Re_1239338.csv",
    delimiter=",",
    skip_header=2,
    usecols=(0, 1, 2, 3, 4, 5, 6),
)
alpha = airfoil_data[:, 0]
cl = airfoil_data[:, 1]
# cd = airfoil_data[:, 2]
# cdp = airfoil_data[:, 3]
# cm = airfoil_data[:, 4]

b = 3.5  # m
W = 25 * 9.81  # N
S = 1.45
AR = (b**2) / S
Lambda = 0  # no idea as no uniform leading edge wing sweep
Rho = 1.225  # kg/m^3
V = 10  # m/s
Cd0 = 0.175
e = 0.85

cd = np.full(cl.shape, Cd0) + ((cl**2) / (np.pi * S * e))

Vd_min = np.sqrt((W / S) * (2 / Rho) * (1 / np.sqrt(Cd0 * e * S * np.pi)))
D_min = 2 * W * np.sqrt((Cd0) / (np.pi * S * e))

Vprmin = 1  # Nothing here yet, need to solve which is in presentation
Pr_min = (
    (4 / 3)
    * W
    * np.sqrt((W / S) * (2 / Rho) * np.sqrt(3 * (Cd0) / ((np.pi * S * e) ** 3)))
)


def CLa_Kuchemann(Cla, Lambda, AR):
    Lambda = np.radians(Lambda)
    num = Cla * np.cos(Lambda)
    dnum = np.sqrt(1 + ((Cla * np.cos(Lambda)) / (np.pi * AR)) ** 2) + (
        (Cla * np.cos(Lambda)) / (np.pi * AR)
    )
    return num / dnum


def CLa_Prandtl(Cla, AR):
    return Cla / (1 + ((Cla) / (np.pi * e * AR)))


def cl_cd_max(cd0, AR):
    return 0.5 * np.sqrt((np.pi * AR * e) / (cd0))


def rd_min(W, S, Rho, cdcl_exp_min, gamma):
    gamma = np.radians(gamma)
    # Reshape cdcl_exp_min to allow broadcasting
    cdcl_exp_min = cdcl_exp_min[:, np.newaxis]
    return np.sqrt((W / S) * (2 / Rho) * cdcl_exp_min * (np.cos(gamma)) ** 3)


def Pa(T, V):
    return T * V


def Pr(D, V):
    return D * V


def Pc(D, T, V):
    return Pa(T, V) - Pr(D, V)


def Vstall(W, Rho, S, Cl):
    return np.sqrt((2 * W) / (Rho * S * Cl))


CLa_Prandtl = CLa_Prandtl(cl, AR)
CLa_Kuchemann = CLa_Kuchemann(cl, Lambda, AR)

alpha_zero_idx = np.abs(alpha).argmin()

cl_array = np.array([cl, CLa_Prandtl, CLa_Kuchemann])
cl_max_idx = np.argmax(cl_array, axis=1)
cl_max = cl_array[:, cl_max_idx]
cl_alpha_zero = cl_array[:, alpha_zero_idx]

cd2_cl3_array = abs((cd**2) / (cl_array**3))
cd2_cl3_min_idx = np.argmin(cd2_cl3_array, axis=1)
cd2_cl3_min = cd2_cl3_array[:, cd2_cl3_min_idx]

# RD Min gamma 0 > 15
gamma = np.arange(0, 15, 0.1)
rd_min_array = np.array([rd_min(W, S, Rho, min_val, gamma) for min_val in cd2_cl3_min])

VstallArray = Vstall(W, Rho, S, cl_max)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# First subplot - RD Min vs Gamma
ax1.plot(gamma, rd_min_array[0][0], label="2D")
ax1.plot(gamma, rd_min_array[1][0], label="Prandtl")
ax1.plot(gamma, rd_min_array[2][0], label="Kuchemann")
ax1.set_xlabel("Gamma (degrees)")
ax1.set_ylabel("RD Min (m/s)")
ax1.set_title("RD Min vs Gamma")
ax1.legend()
ax1.grid()

# Second subplot - CLa vs Angle of Attack
ax2.plot(alpha, CLa_Prandtl, label="CLa Prandtl")
ax2.plot(alpha, CLa_Kuchemann, label="CLa Kuchemann")
ax2.plot(alpha, cl, label="CLa 2D")
ax2.set_xlabel("Angle of Attack (degrees)")
ax2.set_ylabel("CLa")
ax2.set_title("CLa vs Angle of Attack")
ax2.legend()
ax2.grid()

# Third subplot - CD vs Angle of Attack
ax3.plot(alpha, cd, label="CD")
ax3.set_xlabel("Angle of Attack (degrees)")
ax3.set_ylabel("CD")
ax3.set_title("CD vs Angle of Attack")
ax3.legend()
ax3.grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()

print(
    f"CL Max Prandtl: {cl_max[1][0]:.2f} at {alpha[cl_max_idx[1]]} degrees, which results in a stall speed of {VstallArray[1][0]:.2f} m/s"
)
print(
    f"CL Max Kuchemann: {cl_max[2][0]:.2f} at {alpha[cl_max_idx[2]]} degrees, which results in a stall speed of {VstallArray[2][0]:.2f} m/s"
)
print(
    f"CL Max 2D: {cl_max[0][0]:.2f} at {alpha[cl_max_idx[0]]} degrees, which results in a stall speed of {VstallArray[0][0]:.2f} m/s"
)
print(f"Cl/Cd Max: {cl_cd_max(Cd0, AR):.2f}")
print(f"Vd min: {Vd_min:.2f} m/s")
print(f"D min: {D_min:.2f} N")
print(f"Vpr min: {Vprmin} m/s")
print(f"Pr min: {Pr_min:.2f} N")

"""
plt.figure(figsize=(10, 5))
plt.plot(gamma, rd_min_array[0][0], label="2D")
plt.plot(gamma, rd_min_array[1][0], label="Prandtl")
plt.plot(gamma, rd_min_array[2][0], label="Kuchemann")
plt.xlabel("Gamma (degrees)")
plt.ylabel("RD Min (m/s)")
plt.title("RD Min vs Gamma")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(alpha, CLa_Prandtl, label="CLa Prandtl")
plt.plot(alpha, CLa_Kuchemann, label="CLa Kuchemann")
plt.plot(alpha, cl, label="CLa 2D")
plt.xlabel("Angle of Attack (degrees)")
plt.ylabel("CLa")
plt.title("CLa vs Angle of Attack")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(alpha, cd, label="CD")
plt.xlabel("Angle of Attack (degrees)")
plt.ylabel("CD")
plt.title("CD vs Angle of Attack")
plt.legend()
plt.grid()
plt.show()
"""
