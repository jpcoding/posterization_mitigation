import numpy as np
import pandas as pd
import sys

cr_list = []
filenames_list = []

input_path = sys.argv[1]

with open(input_path, 'r') as file:
    for line in file:
        if 'cuszp_cr' in line:
             cr_values =  line.split('=')[1].strip().strip('[]').split(',')
             cr_values = [float(value) for value in cr_values]
             cr_list.append(cr_values)
        if 'filename' in line:
            filename = line.split('=')[1].strip()
            filenames_list.append(filename)

print(filenames_list)
print(cr_list)

flat_cr_list = [cr for sublist in cr_list for cr in sublist]
filename_list = filenames_list
repeated_filenames = []
for filename in filename_list:
    repeated_filenames.extend([filename] * 5)  # Since each filename has 4 CRs

# Create a DataFrame
df = pd.DataFrame({
    'Filename': repeated_filenames,
    'CR': flat_cr_list
})
