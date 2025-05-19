import pandas as pd
import numpy as np
import os

data = pd.read_csv('LOG00002.CSV', delimiter=';')

roll = data[' roll [deg]']
normalized_roll = np.where(
    roll < 0,
    180 - np.abs(roll),
    np.where(roll > 0, roll - 180, roll)
)

def compute_heading(lat1, lon1, lat2, lon2):
    """
    Compute heading (initial bearing) from point 1 to point 2.
    All args in degrees.
    Returns heading in degrees (0 = North, 90 = East).
    """
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    heading = np.degrees(np.arctan2(x, y))
    return (heading + 360) % 360

# Compute heading for each row (except the first, which will be NaN)


data[' Altitude [ft]'] = data[' Approx altitude [m]'] * 3.28084  # Convert meters to feet
data[' Pitch [deg]'] = -1 * (data[' pitch [deg]'] - data[' pitch [deg]'].iloc[0])
data[' Roll [deg]'] = -2*normalized_roll
# Template FDR header (edit as needed)
fdr_header = ["IBM \n","4 \n",
    "ACFT, Aircraft/737NG_Series_U1_XP12/737_80NG.acf\n",
    "TAIL, PH-AE1\n",
    "DATE, 13/05/2025\n",
    "PRES, 29.92\n",
    "DISA, 0\n",
    "WIND, 0,0\n"
]

fdr_dref = [
    "DREF, sim/cockpit2/gauges/indicators/pitch_AHARS_deg_pilot  1.0 \n",
    "DREF, sim/cockpit2/gauges/indicators/roll_AHARS_deg_pilot 1.0\n",
    "DREF, sim/cockpit2/gauges/indicators/altitude_ft_pilot 1.0\n",
    "DREF, sim/cockpit2/gauges/indicators/altitude_ft_copilot 1.0\n",
    "DREF, sim/cockpit2/gauges/indicators/airspeed_kts_pilot 1.0 \n",
    "DREF, sim/cockpit2/gauges/indicators/airspeed_kts_copilot 1.0 \n"
    "DREF, sim/cockpit2/gauges/indicators/heading_AHARS_deg_mag_pilot 1.0 \n",
    "DREF, sim/cockpit2/gauges/indicators/heading_AHARS_deg_mag_copilot 1.0 \n"

    
]

def write_fdr_file(df, output_path):
    with open(output_path, 'w') as f:
        # Write header
        f.writelines(fdr_header)
        f.writelines(fdr_dref)
        line = "COMM, Degrees, Degrees, ft msl, deg, deg, deg\n"
        f.write(line)
        line = "COMM, Time, Longitude, Latitude, Altitude, Heading, Pitch, Roll, Pitch, Roll, Alt, Alt, speed, speed, heading, heading\n"
        f.write(line)
        # Write data lines
        for idx, row in df.iterrows():
            # Example: time, latitude, longitude, altitude, heading, airspeed, etc.
            # Adjust the order and formatting as per X-Plane FDR spec
            line = f"{row[' Time [hh:mm:ss]']}, {row[' Longitude [deg]']:.6f}, {row[' Latitude [deg]']:.6f}, {row[' Altitude [ft]']:.6f}, {row[' Heading [deg]']:.6f}, {row[' Pitch [deg]']:.6f}, {row[' Roll [deg]']:.6f}, {row[' Pitch [deg]']:.6f}, {row[' Roll [deg]']:.6f}, {row[' Altitude [ft]']:.6f}, {row[' Altitude [ft]']:.6f}, {3.6*row[' speed [m/s]']:.6f}, {3.6*row[' speed [m/s]']:.6f}, {row[' Heading [deg]']:.6f}, {row[' Heading [deg]']:.6f}\n"
            f.write(line)

columns_to_average = [
    ' Longitude [deg]',
    ' Latitude [deg]',
    ' Altitude [ft]',
    ' Pitch [deg]',
    ' Roll [deg]',
    ' speed [m/s]'
]



agg_dict = {col: 'mean' for col in columns_to_average}


grouped_data = data.groupby(' Time [hh:mm:ss]', as_index=False).agg(agg_dict)
data = grouped_data.reset_index()

data[' Heading [deg]'] = np.nan
data.loc[1:, ' Heading [deg]'] = compute_heading(
    data[' Latitude [deg]'].iloc[:-1].values,
    data[' Longitude [deg]'].iloc[:-1].values,
    data[' Latitude [deg]'].iloc[1:].values,
    data[' Longitude [deg]'].iloc[1:].values
)
data.loc[0, ' Heading [deg]'] = data.loc[1, ' Heading [deg]']  # Optionally copy the first valid heading

data.loc[:26, ' Heading [deg]'] = data[' Heading [deg]'].iloc[28]
# Example usage:
write_fdr_file(data, 'output.fdr')