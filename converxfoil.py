import pandas as pd
import pandas as pd

# Path to your input file
input_file = "raw_files/bruin_Re_NACA.txt"
output_file = "csv_plots/bruin_Re_NACA.csv"

# Read the file line by line
with open(input_file, "r") as f:
    lines = f.readlines()

# Find the index of the header line (the line with "alpha    CL        CD       CDp       CM")
for i, line in enumerate(lines):
    if line.strip().startswith("alpha"):
        header_line = i
        break

# Use pandas to read the table from the lines
from io import StringIO

# Join only the table part (header + data)
table_text = "".join(lines[header_line:])

# Load into DataFrame
df = pd.read_csv(StringIO(table_text), delim_whitespace=True)

# Save as CSV
df.to_csv(output_file, index=False)

print(f"CSV file saved as: {output_file}")
