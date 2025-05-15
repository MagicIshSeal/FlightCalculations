import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

m = 25
g = 9.81
rho = 1.225
S = 1.45


def normalize_roll(roll):
    if roll < 0:
        return 180 - abs(roll)
    elif roll > 0:
        return roll - 180
    else:
        return roll

raw_data = pd.read_csv('LOG00002.CSV', delimiter=';')
re_v_29 = pd.read_csv('processed_csv/aerodynamics/aero_V_29.2_Re_0.861e6.csv')

processed_data = raw_data.copy()
processed_data['Time[s]'] = (processed_data['Time[ms]'] - processed_data['Time[ms]'].iloc[0])/1000
processed_data[' pitch [deg]'] = -1*(processed_data[' pitch [deg]'] - processed_data[' pitch [deg]'].iloc[0])
processed_data[' roll [deg]'] = processed_data[' roll [deg]'].apply(normalize_roll)
processed_data[' rate of climb [m/s]'] = processed_data[' Approx altitude [m]'].diff() / processed_data['Time[s]'].diff()
processed_data[' rate of climb [m/s] (smoothed)'] = processed_data[' rate of climb [m/s]'].rolling(window=10, center=True).mean()
processed_data[' flight path angle [deg]'] = np.rad2deg(np.asin(processed_data[' rate of climb [m/s] (smoothed)'] / processed_data[' speed [m/s]']))
processed_data[' angle of attack [deg]'] = processed_data[' pitch [deg]'] - processed_data[' flight path angle [deg]']
processed_data[' Lift [N]'] = m*g*(np.cos(np.radians(processed_data[' flight path angle [deg]']))/np.cos(np.radians(abs(processed_data[' roll [deg]']))))
processed_data[' Cl'] = processed_data[' Lift [N]'] / (0.5*rho*(processed_data[' speed [m/s]']**2) *S)

plt.figure()
plt.plot(processed_data['Time[s]'], processed_data[' Approx altitude [m]'], label='Approx altitude [m]')
plt.plot(processed_data['Time[s]'], processed_data[' rate of climb [m/s]'], label='Rate of climb [m/s]')
plt.plot(processed_data['Time[s]'], processed_data[' rate of climb [m/s] (smoothed)'], label='Rate of climb [m/s] (smoothed)')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(processed_data['Time[s]'], processed_data[' flight path angle [deg]'], label='Flight path angle [deg]')
plt.plot(processed_data['Time[s]'], processed_data[' pitch [deg]'], label='pitch [deg]')
plt.legend()
plt.show()

plt.figure()
plt.plot(processed_data['Time[s]'], processed_data[' speed [m/s]'], label='speed[m/s]')
plt.plot(processed_data['Time[s]'], processed_data[' angle of attack [deg]'], label='angle of attack [deg]')
plt.plot(processed_data['Time[s]'], processed_data[' rate of climb [m/s] (smoothed)'], label='rate of climb [m/s] (smoothed)')
plt.plot(processed_data['Time[s]'], processed_data[' Lift [N]'], label='Lift [N]')
plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.plot(processed_data[' angle of attack [deg]'], processed_data[' Cl'],'.', label='Cl')
plt.plot(re_v_29['alpha'], re_v_29['CL_3D'], label='Cl XFoil')
plt.legend()
plt.grid()

plt.show()

        

