import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import folium
from ambiance import Atmosphere

m = 25
g = 9.81
rho = 1.225
S = 1.45

raw_data = pd.read_csv('LOG00002.CSV', delimiter=';')
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

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth (specified in decimal degrees).
    Returns distance in meters.
    """
    R = 6371000  # Earth radius in meters
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat/2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def compute_ground_speed(TAS, heading_deg, wind_speed, wind_dir_deg):
    # Convert angles to radians
    heading_rad = np.deg2rad(heading_deg)
    wind_dir_rad = np.deg2rad(wind_dir_deg)
    
    # TAS vector components
    tas_x = TAS * np.cos(heading_rad)
    tas_y = TAS * np.sin(heading_rad)
    
    # Wind vector components (wind direction is where it's coming FROM, so reverse it)
    wind_x = wind_speed * np.cos(wind_dir_rad + np.pi)
    wind_y = wind_speed * np.sin(wind_dir_rad + np.pi)
    
    # Ground speed vector components
    gs_x = tas_x + wind_x
    gs_y = tas_y + wind_y
    
    # Ground speed magnitude
    ground_speed = np.sqrt(gs_x**2 + gs_y**2)
    # Ground track (direction of motion over ground)
    ground_track_deg = np.rad2deg(np.arctan2(gs_y, gs_x)) % 360
    
    return ground_speed, ground_track_deg

processed_data = raw_data.copy()

roll = processed_data[' roll [deg]']
normalized_roll = np.where(
    roll < 0,
    180 - np.abs(roll),
    np.where(roll > 0, roll - 180, roll)
)


processed_data[' roll [deg]'] = normalized_roll

processed_data['Time[s]'] = (processed_data['Time[ms]'] - processed_data['Time[ms]'].iloc[0])/1000

processed_data[' pitch [deg]'] = -1*(processed_data[' pitch [deg]'] - processed_data[' pitch [deg]'].iloc[0])

processed_data[' rate of climb [m/s]'] = processed_data[' Approx altitude [m]'].diff() / processed_data['Time[s]'].diff()

processed_data[' rate of climb [m/s] (smoothed)'] = processed_data[' rate of climb [m/s]'].rolling(window=10, center=True).mean()

processed_data[' flight path angle [deg]'] = np.rad2deg(np.arcsin(processed_data[' rate of climb [m/s] (smoothed)'] / processed_data[' speed [m/s]']))

processed_data[' angle of attack [deg]'] = processed_data[' pitch [deg]'] - processed_data[' flight path angle [deg]']

processed_data[' accZ [m/s2]'] = processed_data[' accZ [m/s2]'].rolling(window=40, center=True).mean() - g*np.cos(np.deg2rad(processed_data[' pitch [deg]'])) # remove gravity component

processed_data[' Lift [N]'] = m*(processed_data[' accZ [m/s2]']+ np.cos(np.deg2rad(processed_data[ ' pitch [deg]'])*g))

processed_data[' Cl'] = processed_data[' Lift [N]'] / (0.5*rho*(processed_data[' speed [m/s]']**2) *S)
# Interpolate GPS positions between updates to match the logging rate

def gps_fuckery():
    # Only interpolate if there are no NaNs in the GPS columns
    gps_time = processed_data['Time[s]']
    lat_raw = processed_data[' Latitude [deg]']
    lon_raw = processed_data[' Longitude [deg]']

    # Find indices where GPS position changes (i.e., new GPS update)
    gps_change = (lat_raw.diff() != 0) | (lon_raw.diff() != 0)
    gps_change.iloc[0] = True  # Always include the first point

    # Get only the rows where GPS changes
    gps_points = processed_data[gps_change]

    # Interpolate latitude and longitude for all timestamps
    processed_data[' Latitude [deg]'] = np.interp(gps_time, gps_points['Time[s]'], gps_points[' Latitude [deg]'])
    processed_data[' Longitude [deg]'] = np.interp(gps_time, gps_points['Time[s]'], gps_points[' Longitude [deg]'])

    # Now recalculate segment distances and ground speed as before
    distances = [0]
    lats = processed_data[' Latitude [deg]'].values
    lons = processed_data[' Longitude [deg]'].values

    for i in range(1, len(lats)):
        dist = haversine(lats[i-1], lons[i-1], lats[i], lons[i])
        distances.append(dist)

    processed_data[' segment_distance [m]'] = distances
    processed_data[' cumulative_distance [m]'] = processed_data[' segment_distance [m]'].cumsum()
    processed_data[' ground speed [m/s]'] = processed_data[' cumulative_distance [m]'].diff() / processed_data['Time[s]'].diff()
    processed_data[' ground speed [m/s]'] = processed_data[' ground speed [m/s]'].fillna(0)

    # Use .apply to ensure density is a float, not a list or array
    processed_data[' density [kg/m3]'] = processed_data[' Approx altitude [m]'].abs().apply(lambda h: float(np.atleast_1d(Atmosphere(h).density).item()))

    # Now TAS calculation will work as expected
    processed_data[' TAS [m/s]'] = np.sqrt((2 * processed_data[' delta pressure [Pa]']) / processed_data[' density [kg/m3]'])

    gs, track = compute_ground_speed(processed_data[' TAS [m/s]'], processed_data[' heading [deg]'], 6, 45)
    processed_data[' gs_from_TAS [m/s]'] = gs
    processed_data[' gs_track_from_TAS [deg]'] = track  # Optional: store the track as well

    processed_data['speed_err [m/s]'] = abs(processed_data[' gs_from_TAS [m/s]'] - processed_data[' ground speed [m/s]'])

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

processed_data = rotate_points(processed_data, average_angle_ae1, 0, 0, ' angle of attack [deg]', ' Cl')

max_rc_idx = processed_data[' rate of climb [m/s] (smoothed)'].idxmax()
max_rc = processed_data[' rate of climb [m/s] (smoothed)'][max_rc_idx]

print(f'Max rate of climb is {max_rc:.4f} at {processed_data["Time[s]"][max_rc_idx]} seconds.\nPitch is {processed_data[" pitch [deg]"][max_rc_idx]}, FPA is {processed_data[" flight path angle [deg]"][max_rc_idx]:.4f} which means AoA is {abs(processed_data[" angle of attack [deg]"][max_rc_idx]):.4f}.')
print(f'Cl and Alpha uncorrected is, CL ={abs(processed_data[" Cl"][max_rc_idx]):.4f}, AoA = {abs(processed_data[" angle of attack [deg]"][max_rc_idx]):.4f},\nwith correction applied Cl = {abs(processed_data[" Cl_rot"][max_rc_idx]):.4f}, with AoA = {abs(processed_data[" angle of attack [deg]_rot"][max_rc_idx]):.4f}.')
print(f'Airspeed is {processed_data[" speed [m/s]"][max_rc_idx]:.4f} m/s')

check_val_t = 36.520
idx_closest = processed_data['Time[s]'].sub(check_val_t).abs().idxmin()
print("="*100)
print(f'Max rate of climb is {processed_data[" rate of climb [m/s] (smoothed)"][idx_closest]:.4f} at {processed_data["Time[s]"][idx_closest]} seconds.\nPitch is {processed_data[" pitch [deg]"][idx_closest]}, FPA is {processed_data[" flight path angle [deg]"][idx_closest]:.4f} which means AoA is {abs(processed_data[" angle of attack [deg]"][idx_closest]):.4f}.')
print(f'Cl and Alpha uncorrected is, CL ={abs(processed_data[" Cl"][idx_closest]):.4f}, AoA = {abs(processed_data[" angle of attack [deg]"][idx_closest]):.4f},\nwith correction applied Cl = {abs(processed_data[" Cl_rot"][idx_closest]):.4f}, with AoA = {abs(processed_data[" angle of attack [deg]_rot"][idx_closest]):.4f}.')
print(f'Airspeed is {processed_data[" speed [m/s]"][idx_closest]:.4f} m/s')
print("="*100)

ref_time = 150  # to make sure its after takeoff

# Create a mask for rows after the reference time
after_time_mask = processed_data['Time[s]'] > ref_time

# Create a mask for altitude less than 0
alt_below_zero_mask = processed_data[' Approx altitude [m]'] < 0

# Combine masks and find the first index
combined_mask = after_time_mask & alt_below_zero_mask
first_idx = processed_data.index[combined_mask][0] if combined_mask.any() else None

Landing_distance = haversine(processed_data[' Latitude [deg]'][first_idx], processed_data[' Longitude [deg]'][first_idx], processed_data[' Latitude [deg]'].iloc[-1], processed_data[' Longitude [deg]'].iloc[-1])

print(f'Landing distance direct is: {Landing_distance:.2f} m.')

gps_fuckery()

def plot_the_misc():
    plt.figure()
    plt.plot(processed_data['Time[s]'], processed_data[' roll [deg]'], label=r'Roll [$^\circ$]')
    plt.plot(processed_data['Time[s]'], processed_data[' pitch [deg]'], label=r'Pitch [$^\circ$]')
    plt.xlabel(r'Time [$s$]')
    plt.ylabel(r'Angle [$^\circ$]')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(processed_data['Time[s]'], processed_data[' flight path angle [deg]'], label=r'Flight path angle [$^\circ$]')
    plt.plot(processed_data['Time[s]'], processed_data[' pitch [deg]'], label=r'Pitch [$^\circ$]')
    plt.xlabel(r'Time [$s$]')
    plt.ylabel(r'Angle [$^\circ$]')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(processed_data['Time[s]'], processed_data[' speed [m/s]'].rolling(window=10, center=True).mean(), label=r'Speed [$\mathrm{m/s}$]')
    plt.plot(processed_data['Time[s]'], processed_data[' angle of attack [deg]'], label=r'Angle of attack [$^\circ$]')
    plt.plot(processed_data['Time[s]'], processed_data[' rate of climb [m/s] (smoothed)'], label=r'Rate of climb [$\mathrm{m/s}$] (smoothed)')
    plt.xlabel(r'Time [$s$]')
    plt.legend()
    plt.grid()
    plt.show()
    
    plt.figure()
    plt.plot(processed_data[' angle of attack [deg]_rot'], processed_data[' Cl_rot'], '.', label=r'$C_L$ AE1 Inferred AoA')
    plt.plot(re_v_29['alpha'], re_v_29['CL_3D'], label=r'$C_L$ XFoil $V=29.2$')
    plt.plot(re_v_16['alpha'], re_v_16['CL_3D'], label=r'$C_L$ XFoil $V=16.0$')
    plt.plot(re_v_42['alpha'], re_v_42['CL_3D'], label=r'$C_L$ XFoil $V=42.0$')
    plt.xlabel(r'Angle of attack [$^\circ$]')
    plt.ylabel(r'$C_L$')
    plt.title(r'$C_L$ vs Angle of attack')
    plt.legend()
    plt.grid()
    plt.xlim(-20, 30)
    plt.ylim(-2, 2)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(processed_data[' Latitude [deg]'], processed_data[' Longitude [deg]'], processed_data[' Approx altitude [m]'], label='Flight path')
    ax.set_xlabel(r'Latitude [$^\circ$]')
    ax.set_ylabel(r'Longitude [$^\circ$]')
    ax.set_zlabel(r'Approx altitude [$m$]')
    ax.set_title('Flight path in 3D')
    plt.legend()
    plt.show()
    
    
    plt.figure(figsize=(10, 8))
    plt.plot(processed_data[' angle of attack [deg]'], processed_data['speed_err [m/s]'], '.', label='Speed Error [m/s]')
    plt.xlabel('Angle of Attack [$^\circ$]')
    plt.ylabel('Speed Error [m/s]')
    plt.legend()
    plt.grid()
    plt.show()


    plt.figure()
    plt.plot(processed_data['Time[s]'], processed_data[' speed [m/s]'].rolling(window=10, center=True).mean(), label='Speed [m/s] (smoothed)')
    plt.plot(processed_data['Time[s]'], processed_data[' TAS [m/s]'].rolling(window=10, center=True).mean(), label='TAS [m/s] (smoothed)')
    plt.plot(processed_data['Time[s]'], (processed_data[' ground speed [m/s]']).rolling(window=10, center=True).mean(), label='Ground Speed [m/s] (smoothed)')
    plt.plot(processed_data['Time[s]'], processed_data[' gs_from_TAS [m/s]'].rolling(window=10, center=True).mean(), label='Ground Speed from TAS [m/s] (smoothed)')
    plt.ylabel('Speed [m/s]')
    plt.legend()
    plt.grid()
    plt.show()

    # Subplot 1: Speed
    plt.subplot(2, 1, 1)
    plt.plot(processed_data['Time[s]'], processed_data[' speed [m/s]'].rolling(window=10, center=True).mean(), label='Speed [m/s] (smoothed)')
    plt.plot(processed_data['Time[s]'], processed_data[' TAS [m/s]'].rolling(window=10, center=True).mean(), label='TAS [m/s] (smoothed)')
    plt.plot(processed_data['Time[s]'], (processed_data[' ground speed [m/s]'] + processed_data[' speed [m/s]'].iloc[0]).rolling(window=10, center=True).mean(), label='Ground Speed [m/s] (smoothed)')
    plt.plot(processed_data['Time[s]'], processed_data[' gs_from_TAS [m/s]'].rolling(window=10, center=True).mean(), label='Ground Speed from TAS [m/s] (smoothed)')
    plt.ylabel('Speed [m/s]')
    plt.legend()
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(processed_data['Time[s]'], processed_data[' angle of attack [deg]'].rolling(window=10, center=True).mean(), label='Angle of Attack [$^\circ$] (smoothed)')
    plt.xlabel('Time [s]')
    plt.ylabel('Angle of Attack [$^\circ$]')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
    
def plot_landing_run_on_map():
    # Get the landing run coordinates
    lats = processed_data[' Latitude [deg]'].iloc[first_idx:]
    lons = processed_data[' Longitude [deg]'].iloc[first_idx:]

    # Center the map at the start of the landing run
    m = folium.Map(location=[lats.iloc[0], lons.iloc[0]], zoom_start=16)

    # Add the landing run as a polyline
    folium.PolyLine(list(zip(lats, lons)), color='blue', weight=3, opacity=0.8, tooltip="Landing run").add_to(m)

    # Add markers for start and end
    folium.Marker([lats.iloc[0], lons.iloc[0]], popup="Start", icon=folium.Icon(color='green')).add_to(m)
    folium.Marker([lats.iloc[-1], lons.iloc[-1]], popup="End", icon=folium.Icon(color='red')).add_to(m)

    # Display the map
    m.save('landing_run_map.html')
    print("Map saved as landing_run_map.html. Open this file in your browser to view the map.")
    
def plot_the_important():
    #plt.figure()
    #plt.plot(processed_data['Time[s]'], processed_data[' Approx altitude [m]'], label='Approx altitude [$m$]')
    #plt.plot(processed_data['Time[s]'], processed_data[' rate of climb [m/s]'],':', linewidth=1,label=r'Rate of climb [$\frac{m}{s}$]')
    #plt.plot(processed_data['Time[s]'], processed_data[' rate of climb [m/s] (smoothed)'], label=r'Rate of climb [$\frac{m}{s}$] (smoothed)')
    #plt.hlines(max_rc, 0, processed_data['Time[s]'].iloc[-1], linestyles='dashed')
    #plt.xlabel("Time [$s$]")
    #plt.grid()
    #plt.legend()
    #plt.show()
    #plt.figure(figsize=(10, 8))

    # Subplot 1: Altitude
    plt.subplot(2, 1, 1)
    plt.plot(processed_data['Time[s]'], processed_data[' Approx altitude [m]'], label='Approx altitude [$m$]')
    plt.xlabel("Time [$s$]")
    plt.ylabel("Altitude [$m$]")
    plt.grid()
    plt.legend()

    # Subplot 2: Rate of climb
    plt.subplot(2, 1, 2)
    plt.plot(processed_data['Time[s]'], processed_data[' rate of climb [m/s]'],'r:', linewidth=1, label=r'Rate of climb [$\frac{m}{s}$]')
    plt.plot(processed_data['Time[s]'], processed_data[' rate of climb [m/s] (smoothed)'],'g', label=r'Rate of climb [$\frac{m}{s}$] (smoothed)')
    plt.hlines(max_rc, 0, processed_data['Time[s]'].iloc[-1], linestyles='dashed', label='Max rate of climb')
    plt.xlabel("Time [$s$]")
    plt.ylabel(r'Rate of climb [$\frac{m}{s}$]')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    plt.figure()
    plt.plot(processed_data[' Latitude [deg]'].iloc[first_idx:], processed_data[' Longitude [deg]'].iloc[first_idx:], label='Landing run (GPS)')
    plt.plot([processed_data[' Latitude [deg]'].iloc[first_idx],processed_data[' Latitude [deg]'].iloc[-1]],[processed_data[' Longitude [deg]'].iloc[first_idx], processed_data[' Longitude [deg]'].iloc[-1]], label='Calculated Distance')
    plt.xlabel(r'Latitude [$\degree$]')
    plt.ylabel(r'Longtitude [$\degree$]')
    plt.grid()
    plt.legend()
    plt.show()

plot_the_important()
plot_the_misc()



