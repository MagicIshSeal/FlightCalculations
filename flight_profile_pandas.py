import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from XFoil_Runner.Xfoil import run_for_Re
import os


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

V = np.arange(12,42,2)

def Reynolds_number(V, Mu_Air, MAC, Rho):
    return (Rho * V * MAC) / Mu_Air

Re = Reynolds_number(V, Mu_Air, MAC, Rho)
Re = np.array([f"{x/1e6:.3f}e6" for x in Re])

dfs = {} #wrapper for dataframes

for v, re in zip(V, Re):
    dfs[f'df_V_{v}_Re_{re}'] = run_for_Re("NACA2415", re, -18.750, 19.250, 0.25, 1000)

print(dfs)
print(len(dfs))
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

