import matplotlib.pyplot as plt
import os
import polars as pl
import numpy as np
from .globalvars import CP_VAEAC_NN_SIZE_DICT

def get_csv_data(csv_file_name: str) -> pl.DataFrame:
    """Read the csv file into polars DataFrame, keeping the first column as row names."""
    suffix = csv_file_name[-4:]
    csv_file_name += ".csv" if suffix != ".csv" else ""
    save_path = os.path.join('data', csv_file_name)
    return pl.read_csv(save_path, has_header=False)


def plot_data(df: pl.DataFrame, 
              save_name: str,
              avoid_zero_division: bool = False) -> None:
    """Plot the data from the DataFrame."""
    row_names = df[:, 0].to_list() # save names
    df = df.drop(df.columns[0]) # remove names col

    # Normalize each column row-wise, by dividing by the sum of the row.
    row_sums_df = df.sum_horizontal()
    normalized_df = df.with_columns(
            [pl.col(col) / row_sums_df for col in df.columns]
    )

    # If you want to handle cases where row_sums might be zero:
    # normalized_df = df.with_columns(
    #     [pl.when(row_sums != 0).then(pl.col(col) / row_sums_df).otherwise(0) 
    #      for col in df.columns]
    # )

    n_columns = df.width
    n_bars_per_column = df.height - 1  # Assuming first row is for through-lines

    bar_width = 0.2
    x_positions = np.arange(n_columns)

    # Colors and labels
    # TODO: Add more colors for lunar lander and others
    colors = ["skyblue", "salmon", "lightgreen", "gold", "lightcoral"][:n_bars_per_column]
    state_feature_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    bar_labels = row_names[1:]  # Use row names (excluding first row) for bar labels

    # Plot bars for each column
    for col_idx in range(n_columns):
        for bar_idx in range(n_bars_per_column):
            x = x_positions[col_idx] + (bar_idx - (n_bars_per_column-1)/2) * bar_width
            value = normalized_df.row(bar_idx + 1)[col_idx]  # +1 to skip through-line row/col
            plt.bar(x, value, width=bar_width, color=colors[bar_idx], 
                        label=bar_labels[bar_idx] if col_idx == 0 else "")

    # Plot through-lines
    for col_idx in range(n_columns):
        line_value = normalized_df.row(0)[col_idx]
        x_min = x_positions[col_idx] - (n_bars_per_column/2) * bar_width
        x_max = x_positions[col_idx] + (n_bars_per_column/2) * bar_width
        plt.hlines(line_value, x_min, x_max, colors="red", linestyles="dashed", 
                 label="Policy prediction" if col_idx == 0 else "")

    # Make a check for min val in normalized_df
    normalized_df_min = normalized_df.min().to_numpy().min()

    # Plot configs
    y_min = normalized_df_min if normalized_df_min < -0.2 else -0.2
    if y_min < -0.2:
        print(f"WARNING: y_min is set to {y_min}, other plots need to be adjusted accordingly.")
    plt.ylim(y_min, 1.)
    plt.xticks(x_positions, labels=state_feature_names)
    plt.axhline(0, color="black", linewidth=0.5)
    plt.ylabel("Normalized Shapley Value")
    plt.title("Shapley Values for Cartpole State Features")
    plt.legend(loc="upper left")
    plt.grid()
    plt.tight_layout()

    # Save the plot
    if not os.path.exists("plots"):
        os.makedirs("plots")
    save_path = os.path.join("plots", save_name)
    plt.savefig(save_path)
    plt.clf()


def plot_data_from_id(csv_filename:str,
                      plot_save_name: str,
                      avoid_zero_div: bool = False
                      ) -> None:
    """
    Uses helper functions plotting.plot_data and plotting.get_csv_data,
    to plot data from csv files. ID is contained in the csv filename.
    """
    # Regex to ofilter the id at the end of csv_filename
    df = get_csv_data(csv_filename)
    plot_data(df, plot_save_name, avoid_zero_division=avoid_zero_div) 


def readjust_plots_from_data(verbose=True):
    """
    Remakes plots from all csv files in the 'data' directory, 
    which are run with the exp4 setup.
    """
    for csv_file in os.listdir("data"):
        if "trajectory" in csv_file:
            continue
        if csv_file.endswith(".csv"):
            # Extract the ID from the filename
            id_part = csv_file.split("_")[-1].replace(".csv", "")
            exp_num = int(csv_file.split("_")[2])
            env = csv_file.split("_")[0]
            env = "CP" if env == "cartpole" else env
            sd = {} # size_dict
            for key in CP_VAEAC_NN_SIZE_DICT.keys():
                sd[key] = CP_VAEAC_NN_SIZE_DICT[key][exp_num - 1]
            save_name = f"{env}_{exp_num}_LD_{sd['latent_dim']}_W_{sd['width']}_D_{sd['depth']}_{id_part}"
            plot_data_from_id(csv_file, save_name)
            if verbose:
                print(f"Plot saved successfully at 'plots/{save_name}.png'")

if __name__ == "__main__":
    readjust_plots_from_data(verbose=True)

