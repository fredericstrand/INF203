import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def box_average(
    path, block_size=50, z=1.96, skiprow=12, plot=False, wanted_col="E_pot"
):
    """
    Calculates the block-averaged mean and confidence interval from simulation data

    ...

    Parameters
    ----------
    path : str
        Path to the log file containing data
    block_size : int, optional
        Number of samples per block for averaging (default is 50)
    z : float, optional
        Z-score for desired confidence interval (default is 1.96, corresponding to 95%)
    skiprow : int, optional
        Number of lines to skip before the table starts (default is 12)
    plot : bool, optional
        Whether to show a histogram of block means (default is False)

    Returns
    -------
    mean : float
        The total average mean of the blocks of the wanted column
    ci : float
        confidence interval for the mean and z value
    """

    with open(path, "r") as f:
        lines = f.readlines()
    column_names = lines[skiprow - 1].strip("#").strip().split()

    df = pd.read_csv(
        path, delim_whitespace=True, skiprows=skiprow, names=column_names, comment="#"
    )

    data = df[wanted_col].to_numpy()
    n_blocks = len(data) // block_size
    blocks = data[: n_blocks * block_size].reshape(n_blocks, block_size)

    means = blocks.mean(axis=1)
    mean = means.mean()
    stde = means.std(ddof=1) / np.sqrt(n_blocks)
    ci = z * stde
    if plot:
        plt.hist(means, bins=50, label="Block means")

    return mean, ci
