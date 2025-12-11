import os
import subprocess
import numpy as np
import pandas as pd


def run_xfoil(
    airfoil_name, alpha_start, alpha_end, Re, alpha_step=0.25, Ncrit=9, n_iter=100
):
    # Define base directory and output directory
    base_dir = "XFoil_Runner"
    output_dir = os.path.join(base_dir, "output_polars")

    # Create polar filename
    polar_file = os.path.join(output_dir, f"polar_file_{airfoil_name}_Re_{Re}.txt")

    # Check and remove existing file
    if os.path.exists(polar_file):
        os.remove(polar_file)

    # Create input file path
    input_file_path = os.path.join(base_dir, "input_file.in")

    # Write input file
    with open(input_file_path, "w") as input_file:
        input_file.write(f"{airfoil_name}\n")
        input_file.write("PANE\n")
        input_file.write("OPER\n")
        input_file.write("VPAR\n")
        input_file.write(f"N {Ncrit}\n")
        input_file.write("\n")
        input_file.write(f"Visc {Re}\n")
        input_file.write("PACC\n")
        input_file.write(f"{polar_file}\n\n")
        input_file.write(f"ITER {n_iter}\n")
        input_file.write(f"ASeq {alpha_start} {alpha_end} {alpha_step}\n")
        input_file.write("\n\n")
        input_file.write("quit\n")

    # Run XFoil with platform-specific command
    if os.name == "nt":  # Windows
        xfoil_exe = os.path.join(base_dir, "xfoil.exe")
        subprocess.call(f"{xfoil_exe} < {input_file_path}", shell=True)
    else:  # Linux/Unix
        subprocess.call(f"xfoil < {input_file_path}", shell=True)


def format_polar_data_pandas(polar_file):
    """Format the polar data for easier analysis"""
    try:
        # Read the data and see what columns we actually have
        polar_df = pd.read_csv(
            polar_file,
            delim_whitespace=True,
            skiprows=12,
            usecols=(0, 1, 2, 3, 4, 5, 6),
        )
        print(f"Columns found in file: {polar_df.columns}")

        # Assign columns based on what's actually in the file
        column_names = ["alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"]
        polar_df.columns = column_names[: len(polar_df.columns)]

        return polar_df
    except Exception as e:
        print(f"Error reading file: {e}")
        print(f"File contents:\n{open(polar_file).read()}")
        raise


def run_for_Re(
    airfoil_name, alpha_start, alpha_end, Re, alpha_step=0.25, Ncrit=9, n_iter=100
):
    """Run XFoil for a given airfoil and Re number"""
    run_xfoil(
        airfoil_name,
        alpha_start,
        alpha_end,
        Re,
        alpha_step=alpha_step,
        Ncrit=Ncrit,
        n_iter=n_iter,
    )

    # Create polar file path
    polar_file = os.path.join(
        "XFoil_Runner", "output_polars", f"polar_file_{airfoil_name}_Re_{Re}.txt"
    )

    polar_df = format_polar_data_pandas(polar_file)
    return polar_df


if __name__ == "__main__":
    airfoil_name = "NACA2415"
    alpha_start = -20
    alpha_end = 25
    Re = 1e6
    alpha_step = 0.5
    n_iter = 100
    ncrit = 5

    df = run_for_Re(
        airfoil_name,
        alpha_start,
        alpha_end,
        Re,
        alpha_step=alpha_step,
        Ncrit=ncrit,
        n_iter=n_iter,
    )
    
    # Save to CSV with alpha, CL, CD columns
    output_csv = os.path.join("XFoil_Runner", "output_csv", f"Re_{Re:.3e}.csv")
    df[["alpha", "CL", "CD"]].to_csv(output_csv, index=False)
    print(f"Saved output to {output_csv}")
    
    
