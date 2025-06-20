from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def box_average(
    path: str,
    z: Optional[float] = 1.96,
    skiprow: int = 12,
    plot: Optional[bool] = True,
    wanted_col: str = "E_pot",
):
    """
    Calculates the mean and confidence interval given 1 line=1 "block" as file is already only done every 50 or so.

    Parameters
    ----------
    path : str
        Path to the log file containing data
    z : float, optional
        Z-score for desired confidence interval (default is 1.96, 95%).
    skiprow : int
        Number of lines to skip before the table starts (default is 12)
    plot : bool, optional
        Whether to show a histogram of the values (default is False)
    wanted_col : str
        Name of the column to analyze (default is 'E_pot')

    Returns
    -------
    mean : float
        Mean of the selected column
    ci : float
       confidence interval
    """
    with open(path, "r") as f:
        lines = f.readlines()
    if len(lines) <= skiprow:
        raise ValueError(f"Not enough lines in file '{path}' to skip {skiprow} rows")
    column_names = lines[skiprow - 1].strip("#").strip().split()

    df = pd.read_csv(path, sep="\s+", skiprows=skiprow, names=column_names, comment="#")

    if wanted_col not in df.columns:
        raise ValueError(f"Column '{wanted_col}' not found in file '{path}'")

    data = pd.to_numeric(df[wanted_col], errors="coerce").dropna().to_numpy()
    mean = float(np.mean(data))
    stde = np.std(data) / np.sqrt(len(data))
    ci = z * stde

    if plot:
        plt.hist(data, bins=len(data))
        plt.axvline(
            mean,
            color="red",
            linestyle="--",
        )
        plt.axvline(mean - ci, color="gray", linestyle="--")
        plt.axvline(mean + ci, color="gray", linestyle="--")
        plt.show()

    return mean, ci
