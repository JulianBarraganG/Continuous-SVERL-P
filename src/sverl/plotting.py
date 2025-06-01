import matplotlib.pyplot as plt
import os
import polars as pl
import numpy as np

def get_csv_data(csv_file_name: str) -> pl.DataFrame:
    """Read the csv file into polars DataFrame, keeping the first column as row names."""
    suffix = csv_file_name[-4:-1]
    csv_file_name += ".csv" if suffix != ".csv" else csv_file_name
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
    colors = ['skyblue', 'salmon', 'lightgreen', 'gold', 'lightcoral'][:n_bars_per_column]
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
        line_value = row_names[col_idx]
        x_min = x_positions[col_idx] - (n_bars_per_column/2) * bar_width
        x_max = x_positions[col_idx] + (n_bars_per_column/2) * bar_width
        plt.hlines(line_value, x_min, x_max, colors='red', linestyles='dashed', 
                 label='Ground truth' if col_idx == 0 else "")

    # Plot configs
    plt.xticks(x_positions, labels=state_feature_names)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.ylabel('Shapley Value')
    plt.title('Shapley Values with Reference Lines')
    plt.legend(loc='upper left')
    plt.grid()
    plt.tight_layout()

    # Save the plot
    if not os.path.exists('plots'):
        os.makedirs('plots')
    save_path = os.path.join('plots', save_name)
    plt.savefig(save_path)

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



if __name__ == "__main__":
    # Plot the data
    df = get_csv_data(os.path.join('data', 'cartpole_shapley_values.csv'))
    # df = get_csv_data(os.path.join('data', 'cartpole_shapley_values_27-05.csv'))
    plot_data(df, 'shapley_values_plot.png')
    print("Plot saved successfully.")
