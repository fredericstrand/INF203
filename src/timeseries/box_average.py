import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def box_average(path, block_size=50, z=1.96, skiprow=0, plot=False):
    with open(path, "r") as f:
        lines = f.readlines()
    column_names = lines[skiprow - 1].strip("#").strip().split()

    df = pd.read_csv(
        path, delim_whitespace=True, skiprows=skiprow, names=column_names, comment="#"
    )

    data = df["E_pot"].to_numpy()
    n_blocks = len(data) // block_size
    blocks = data[: n_blocks * block_size].reshape(n_blocks, block_size)

    means = blocks.mean(axis=1)
    mean = means.mean()
    stde = means.std(ddof=1) / np.sqrt(n_blocks)
    ci = z * stde
    if plot:
        plt.hist(means, bins=50, label="Block means")

    return mean, ci
