import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d

m = 25
g = 9.81
rho = 1.225
S = 1.45


raw_data = pd.read_csv('LOG00002.CSV', delimiter=';')
ae4  = pd.read_csv('AE4.CSV', delimiter=';', skiprows=range(1, 550))
re_v_29 = pd.read_csv('processed_csv/aerodynamics/aero_V_29.2_Re_0.861e6.csv')
re_v_16 = pd.read_csv('processed_csv/aerodynamics/aero_V_16.0_Re_0.472e6.csv')
re_v_42 = pd.read_csv('processed_csv/aerodynamics/aero_V_42.0_Re_1.239e6.csv')

def rotate_points(df, angle_deg, cx, cy, x_col, y_col):
    """
    Rotates points in df[x_col], df[y_col] by angle_deg degrees around (cx, cy).
    Adds new columns: x_col+'_rot', y_col+'_rot'.
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    x_shifted = df[x_col] - cx
    y_shifted = df[y_col] - cy
    df[x_col + '_rot'] = cos_theta * x_shifted - sin_theta * y_shifted + cx
    df[y_col + '_rot'] = sin_theta * x_shifted + cos_theta * y_shifted + cy
    return df

def rotation_loss(angle_deg, df, theory_x, theory_y, x_col, y_col):
    # Rotate points
    rotated = rotate_points(df.copy(), angle_deg, 0, 0, x_col, y_col)
    x_rot = rotated[x_col + '_rot']
    y_rot = rotated[y_col + '_rot']
    # Interpolate theory curve at rotated x positions
    interp_theory = interp1d(theory_x, theory_y, bounds_error=False, fill_value='extrapolate')
    theory_y_interp = interp_theory(x_rot)
    # Compute sum of squared differences
    return np.sum((y_rot - theory_y_interp) ** 2)



processed_data = raw_data.copy()

roll = processed_data[' roll [deg]']
normalized_roll = np.where(
    roll < 0,
    180 - np.abs(roll),
    np.where(roll > 0, roll - 180, roll)
)


processed_data[' roll [deg]'] = normalized_roll

ae4['Time[s]'] = (ae4['Time[ms]'] - ae4['Time[ms]'].iloc[0])/1000
processed_data['Time[s]'] = (processed_data['Time[ms]'] - processed_data['Time[ms]'].iloc[0])/1000

ae4[' pitch [deg]'] = -1*(ae4[' pitch [deg]'] - ae4[' pitch [deg]'].iloc[0])
processed_data[' pitch [deg]'] = -1*(processed_data[' pitch [deg]'] - processed_data[' pitch [deg]'].iloc[0])

ae4[' rate of climb [m/s]'] = ae4[' Approx altitude [m]'].diff() / ae4['Time[s]'].diff()
processed_data[' rate of climb [m/s]'] = processed_data[' Approx altitude [m]'].diff() / processed_data['Time[s]'].diff()

ae4['rate of climb [m/s] (smoothed)'] = ae4[' rate of climb [m/s]'].rolling(window=10, center=True).mean()
processed_data[' rate of climb [m/s] (smoothed)'] = processed_data[' rate of climb [m/s]'].rolling(window=10, center=True).mean()

ae4[' flight path angle [deg]'] = np.rad2deg(np.asin(ae4['rate of climb [m/s] (smoothed)'] / ae4[' speed [m/s]']))
processed_data[' flight path angle [deg]'] = np.rad2deg(np.asin(processed_data[' rate of climb [m/s] (smoothed)'] / processed_data[' speed [m/s]']))

ae4[' angle of attack [deg]'] = ae4[' pitch [deg]'] - ae4[' flight path angle [deg]']
processed_data[' angle of attack [deg]'] = processed_data[' pitch [deg]'] - processed_data[' flight path angle [deg]']

ae4[' accz [m/s2]'] = ae4[' accZ [m/s2]'].rolling(window=40, center=True).mean() - g
processed_data[' accZ [m/s2]'] = processed_data[' accZ [m/s2]'].rolling(window=40, center=True).mean() - g

ae4[' Lift [N]'] = m*(ae4[' accZ [m/s2]'] + np.cos(np.deg2rad(ae4[' pitch [deg]'])*g))
processed_data[' Lift [N]'] = m*(processed_data[' accZ [m/s2]']+ np.cos(np.deg2rad(processed_data[ ' pitch [deg]'])*g))

ae4[' Cl'] = ae4[' Lift [N]'] / (0.5*rho*(ae4[' speed [m/s]']**2) *S)
processed_data[' Cl'] = processed_data[' Lift [N]'] / (0.5*rho*(processed_data[' speed [m/s]']**2) *S)

dfs = [re_v_29, re_v_16, re_v_42]
angles_ae1 = []
for df in dfs:
    theory_x = df['alpha']
    theory_y = df['CL_3D']
    res = minimize(
        rotation_loss,
        x0=0,  # initial guess for angle
        args=(processed_data, theory_x, theory_y, ' angle of attack [deg]', ' Cl'),
        bounds=None  # restrict search if desired
        )
    angles_ae1.append(res.x[0])
average_angle_ae1 = np.average(angles_ae1)
print(f"average best rotation angle for AE1 inferred: {average_angle_ae1:.2f} degrees")

angles_ae4_inferred = []
for df in dfs:
    theory_x = df['alpha']
    theory_y = df['CL_3D']
    res = minimize(
        rotation_loss,
        x0=0,  # initial guess for angle
        args=(ae4, theory_x, theory_y, ' angle of attack [deg]', ' Cl'),
        bounds=None  # restrict search if desired
        )
    angles_ae4_inferred.append(res.x[0])
average_angle_ae4_inferred = np.average(angles_ae4_inferred)
print(f"average best rotation angle for AE4 inferred: {average_angle_ae4_inferred:.2f} degrees")

angles_ae4_data = []
for df in dfs:
    theory_x = df['alpha']
    theory_y = df['CL_3D']
    res = minimize(
        rotation_loss,
        x0=0,  # initial guess for angle
        args=(ae4, theory_x, theory_y, ' Angle [deg]', ' Cl'),
        bounds=None  # restrict search if desired
        )
    angles_ae4_data.append(res.x[0])
average_angle_ae4_data = np.average(angles_ae4_data)
print(f"average best rotation angle for AE4 data: {average_angle_ae4_data:.2f} degrees")

processed_data = rotate_points(processed_data, average_angle_ae1, 0, 0, ' angle of attack [deg]', ' Cl')
ae4 = rotate_points(ae4, average_angle_ae4_inferred, 0, 0, ' angle of attack [deg]', ' Cl')
ae4 = rotate_points(ae4, average_angle_ae4_data, 0, 0, ' Angle [deg]', ' Cl')

processed_data[' Cl_rot_trans'] = processed_data[' Cl_rot'] +0.25


plt.figure()
plt.plot(processed_data['Time[s]'], processed_data[' roll [deg]'], label='roll [deg]')
plt.plot(processed_data['Time[s]'], processed_data[' pitch [deg]'], label='pitch [deg]')
plt.legend()
plt.grid()
plt.show()

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
#plt.plot(processed_data['Time[s]'], processed_data[' Lift [N]'], label='Lift [N]')
plt.grid()
plt.legend()
plt.show()

ae4[' Angle [deg]'] = ae4[' Angle [deg]'] - ae4[' Angle [deg]'].iloc[0]

plt.figure()
plt.plot(processed_data[' angle of attack [deg]_rot'], processed_data[' Cl_rot'],'.', label='Cl AE1 Inferred AoA')
plt.plot(ae4[' angle of attack [deg]_rot'], ae4[' Cl_rot'],'.', label='Cl AE4 Inferred AoA')
plt.plot(ae4[' Angle [deg]_rot'], ae4[' Cl_rot'],'.', label='Cl AE4 Data AoA')
plt.plot(re_v_29['alpha'], re_v_29['CL_3D'], label='Cl XFoil V=29.2')
plt.plot(re_v_16['alpha'], re_v_16['CL_3D'], label='Cl XFoil V=16.0')
plt.plot(re_v_42['alpha'], re_v_42['CL_3D'], label='Cl XFoil V=42.0')
plt.xlabel('Angle of attack [deg]')
plt.ylabel('Cl')
plt.title('Cl vs Angle of attack')
plt.legend()
plt.grid()
plt.xlim(-20,30)
plt.ylim(-2,2)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(processed_data[' Latitude [deg]'], processed_data[' Longitude [deg]'], processed_data[' Approx altitude [m]'], label='Flight path')
ax.set_xlabel('Latitude [deg]')
ax.set_ylabel('Longitude [deg]')
ax.set_zlabel('Approx altitude [m]')
ax.set_title('Flight path in 3D')
plt.legend()
plt.show()

        

