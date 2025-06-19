import pandas as pd
import numpy as np
import os
from sverl.globalvars import CP_STATE_FEATURE_NAMES

filenames = []
for i, csv_file in enumerate(os.listdir("data")):
    if csv_file.endswith(".csv") and "250605" not in csv_file and "trajectory" not in csv_file:
        filenames.append(csv_file)

data = np.zeros((4,4,5))

for i, filename in enumerate(filenames):
    this_data = np.loadtxt(f"data/{filename}", delimiter=",", usecols=range(1, 5), skiprows=1)
    # Normalize such that each this_data row sums to 1
    row_sums = this_data.sum(axis=1, keepdims=True)
    this_data = this_data / row_sums
    data[:,:,i] = this_data


means = np.mean(data, axis=2) 
stds = np.std(data, axis=2)


#print(pl.DataFrame(means, schema=CP_STATE_FEATURE_NAMES).to_pandas().to_latex(index=False, float_format="%.2f"))
#print(pl.DataFrame(stds, schema=CP_STATE_FEATURE_NAMES).to_pandas().to_latex(index=False, float_format="%.2f"))
mean_latex = pd.DataFrame(means, columns=CP_STATE_FEATURE_NAMES).to_latex(index=False, float_format="%.4f")
std_latex = pd.DataFrame(stds, columns=CP_STATE_FEATURE_NAMES).to_latex(index=False, float_format="%.4f")
print(mean_latex)

