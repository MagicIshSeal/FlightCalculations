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

V = np.arange(12,42,2)

def Reynolds_number(V, Mu_Air, MAC, Rho):
    return (Rho * V * MAC) / Mu_Air

Re = Reynolds_number(V, Mu_Air, MAC, Rho)
Re = np.array([f"{x/1e6:.3f}e6" for x in Re])

dfs = {} #wrapper for dataframes

choice = input("Do you want to run XFoil? (y/n): ").strip().lower()

if choice == 'y':
    for v, re in zip(V, Re):
        dfs[f'df_V_{v}_Re_{re}'] = run_for_Re("NACA2415", re, -18.750, 19.250, 0.25, 1000)
    with open('dfs_dictionary.pkl', 'wb') as file:
        pd.to_pickle(dfs, file)
elif choice == 'n':
    with open('dfs_dictionary.pkl', 'rb') as file:
        dfs = pd.read_pickle(file)

print(dfs)

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

for df_name, df in dfs.items():
    # Add the calculated values to the DataFrame
    df['CLa_prandtl'] = CLa_Prandtl(df['CL'], AR)
    df['CLa_kuchemann'] = CLa_Kuchemann(df['CL'], Lambda, AR)

    dfs[df_name] = df

print(dfs[list(dfs.keys())[0]]['CLa_prandtl'])  # Print first dataframe's Prandtl results
# Create 5 subplots (2 rows, 3 columns with last spot empty)
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])

fig.suptitle('Aerodynamic Characteristics at Different Velocities', fontsize=16)

for i, (df_name, df) in enumerate(dfs.items()):
    velocity = V[i]  # Get corresponding velocity
    
    # Plot CL vs alpha
    ax1.plot(df['alpha'], df['CL'], label=f'V={velocity} m/s')
    ax1.set_xlabel('Angle of Attack (degrees)')
    ax1.set_ylabel('Lift Coefficient (CL)')
    ax1.set_title('CL vs Alpha')
    ax1.grid(True)
    
    # Plot CD vs alpha
    ax2.plot(df['alpha'], df['CD'], label=f'V={velocity} m/s')
    ax2.set_xlabel('Angle of Attack (degrees)')
    ax2.set_ylabel('Drag Coefficient (CD)')
    ax2.set_title('CD vs Alpha')
    ax2.grid(True)
    
    # Plot CL vs CD (drag polar)
    ax3.plot(df['CD'], df['CL'], label=f'V={velocity} m/s')
    ax3.set_xlabel('Drag Coefficient (CD)')
    ax3.set_ylabel('Lift Coefficient (CL)')
    ax3.set_title('Drag Polar')
    ax3.grid(True)
    
    # Plot CLa Prandtl
    ax4.plot(df['alpha'], df['CLa_prandtl'], label=f'V={velocity} m/s')
    ax4.set_xlabel('Angle of Attack (degrees)')
    ax4.set_ylabel('CLa Prandtl')
    ax4.set_title('CLa Prandtl vs Alpha')
    ax4.grid(True)
    
    # Plot CLa Kuchemann
    ax5.plot(df['alpha'], df['CLa_kuchemann'], label=f'V={velocity} m/s')
    ax5.set_xlabel('Angle of Attack (degrees)')
    ax5.set_ylabel('CLa Kuchemann')
    ax5.set_title('CLa Kuchemann vs Alpha')
    ax5.grid(True)

# Add legends to all subplots
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()