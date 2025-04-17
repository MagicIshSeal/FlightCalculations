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
    "csv_plots/bruin_Re_NACA.csv",
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
T = 160  # N
V = np.arange(10, 42, 0.1)
Cd0 = 0.175
e = 0.85

cd = np.full(cl.shape, Cd0) + ((cl**2) / (np.pi * AR * e))

Vd_min = np.sqrt((W / S) * (2 / Rho) * (1 / np.sqrt(Cd0 * e * AR * np.pi)))
D_min = 2 * W * np.sqrt((Cd0) / (np.pi * AR * e))

Vprmin = 1  # Nothing here yet, need to solve which is in presentation
Pr_min = (
    (4 / 3)
    * W
    * np.sqrt((W / S) * (2 / Rho) * np.sqrt(3 * (Cd0) / ((np.pi * AR * e) ** 3)))
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


def Pa(V):
    return T * V


def Pr(V, Cl):
    D = (Cd0 * 0.5 * Rho * S * (V**2)) + ((Cl**2) / (np.pi * AR * e) * 0.5 * Rho * S)
    return D * V


def Pc(V, Cl):
    D = (Cd0 * 0.5 * Rho * S * (V**2)) + ((Cl**2) / (np.pi * AR * e) * 0.5 * Rho * S)
    return (T * V) - (D * V)


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
ax2.plot(alpha, CLa_Prandtl, label="$C_{L_{\\alpha}}$ Prandtl")
ax2.plot(alpha, CLa_Kuchemann, label="$C_{L_{\\alpha}}$ Kuchemann")
ax2.plot(alpha, cl, label="$C_{L_{\\alpha}}$ 2D")
ax2.axhline(
    y=cl_max[0][0], color="blue", linestyle="--", alpha=0.5, label="$C_{L_{max}}$ 2D"
)
ax2.axhline(
    y=cl_max[1][0],
    color="orange",
    linestyle="--",
    alpha=0.5,
    label="$C_{L_{max}}$ Prandtl",
)
ax2.axhline(
    y=cl_max[2][0],
    color="green",
    linestyle="--",
    alpha=0.5,
    label="$C_{L_{max}}$ Kuchemann",
)

ax2.set_xlabel("Angle of Attack (degrees)")
ax2.set_ylabel("$C_{L_{\\alpha}}$")
ax2.set_title("$C_{L_{\\alpha}}$ vs Angle of Attack")
ax2.legend()
ax2.grid()

# Third subplot - CD vs Angle of Attack
ax3.plot(alpha, cd, label="$C_D$")
ax3.set_xlabel("Angle of Attack (degrees)")
ax3.set_ylabel("$C_D$")
ax3.set_title("$C_D$ vs Angle of Attack")
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


# Pa_array = Pa(V)
# Pr_array_2d = np.array([Pr(V, Cl) for Cl in cl_array[0]])
# Pr_array_Prandel = np.array([Pr(V, Cl) for Cl in cl_array[1]])
# Pr_array_Kuchemann = np.array([Pr(V, Cl) for Cl in cl_array[2]])
## Pc_array = Pc(V, cl_array)
#
# print(np.shape(Pr_array_2d))
# print(np.shape(V))
# print(np.shape(cl_array[0]))
# plt.figure()
# for i in range(len(cl_array[0])):
#    plt.plot(V, Pr_array_2d[i], label="Pr 2D")
#    plt.plot(V, Pr_array_Prandel[i], label="Pr Prandtl")
#
# plt.xlabel("Velocity (m/s)")
# plt.ylabel("Power Required (W)")
# plt.title("Power Required vs Velocity")
# plt.legend()
# plt.grid()
# plt.show()
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
