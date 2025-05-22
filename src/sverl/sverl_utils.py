import csv
import numpy as np
import time
import os
from tqdm import trange
from gymnasium import Env

from .shapley_utils import shapley_value, global_sverl_value_function

def report_sverl_p(shapley_values: np.ndarray,
                    state_feature_names: list[str],
                   data_file_prefix: str = "",
                    row_name: str | None = None,
                   verbose: bool = True) -> None:
    """
    Report the Shapley values of the state features and the value of the empty set.

    Parameters
    ----------
    shapley_values : list
        Shapley values for each state feature.
    state_feature_names : list[str]
        Names of the state features.
    """
    data_file_name = data_file_prefix + "_shapley_values.csv"

    # Print the Shapley values.
    if verbose:
        right_adjust = max([(len(f_name)) for f_name in state_feature_names]) + 1
        for i, shap_vl in enumerate(shapley_values):
            print(f"Shapley value of {state_feature_names[i]:<{right_adjust}}: {shap_vl:>8.2f}")

    # Write to csv file.
    if row_name is not None:
        if data_file_prefix == "":
            raise ValueError("data_file_prefix must be specified if row_name is provided.")
        if not os.path.exists("data"):
            os.makedirs("data")
        file_path = os.path.join("data", data_file_name)
        row = [row_name] + shapley_values.tolist()
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

