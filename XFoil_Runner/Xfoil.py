import os
import subprocess
import numpy as np
import pandas as pd



def run_xfoil(airfoil_name, alpha_start, alpha_end, alpha_step, Re, n_iter):
    if os.path.exists(f"XFoil_Runner\output_polars\polar_file_{airfoil_name}_Re_{Re}.txt"):
        os.remove(f"XFoil_Runner\output_polars\polar_file_{airfoil_name}_Re_{Re}.txt")
    
    input_file = open(r"XFoil_Runner\input_file.in", 'w')
    input_file.write(airfoil_name + '\n')
    input_file.write("PANE\n")
    input_file.write("OPER\n")
    input_file.write("Visc {0}\n".format(Re))
    input_file.write("PACC\n")
    input_file.write(f"XFoil_Runner\output_polars\polar_file_{airfoil_name}_Re_{Re}.txt\n\n")
    input_file.write("ITER {0}\n".format(n_iter))
    input_file.write("ASeq {0} {1} {2}\n".format(alpha_start, alpha_end,
                                                 alpha_step))
    input_file.write("\n\n")
    input_file.write("quit\n")
    input_file.close()

    subprocess.call(r"XFoil_Runner\xfoil.exe < XFoil_Runner\input_file.in", shell=True)

def format_polar_data_pandas(polar_file):
    """Format the polar data for easier analysis"""
    # Skip the first 12 rows and convert to a DataFrame
    polar_df = pd.read_csv(polar_file, delim_whitespace=True, skiprows=12, header=None)
    polar_df.columns = ["alpha", "CL", "CD","CDp", "CM", "Top_Xtr", "Bot_Xtr"]

    #polar_df.to_csv(f"XFoil_Runner\output_csv\Re_{Re}.csv", index=False)
    return polar_df

def run_for_Re(airfoil_name, Re, alpha_start, alpha_end, alpha_step, n_iter):
    """Run XFoil for a given airfoil and Re number"""
    run_xfoil(airfoil_name, alpha_start, alpha_end, alpha_step, Re, n_iter)
    polar_file = f"XFoil_Runner\output_polars\polar_file_{airfoil_name}_Re_{Re}.txt"
    polar_df = format_polar_data_pandas(polar_file)
    return polar_df