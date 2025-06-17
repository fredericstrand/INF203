import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Union

def _parse_ta_log(lines: Sequence[str]):
    """
    Extract step, potential energy, gamma1, gamma2 from a TA-method log.
    Returns 4 numpy arrays of equal length.
    """
    data = []
    pattern = re.compile(
        r"^\s*(\d+)\s+"        # step
        r"([-+]?\d+\.\d+)\s+"  # E_pot
        r"\S+\s+\S+\s+\S+\s+"  # skip acc, e1
        r"([-+]?\d+\.\d+)\s+"  # avg1
        r"([-+]?\d+\.\d+)"     # gamma1
        r".*?([-+]?\d+\.\d+)$" # gamma2
    )
    for ln in lines:
        m = pattern.match(ln)
        if m:
            step, epot, _, g1, g2 = m.groups()
            data.append((int(step), float(epot), float(g1), float(g2)))
    if not data:
        raise ValueError("No data lines recognised – check log format")
    return (np.array(col) for col in zip(*data))

def plot_energy_and_gamma(log: Union[str, Path, Sequence[str]]):

    if isinstance(log, (str, Path)):
        with open(log, "r") as fh:
            lines = fh.readlines()
    else:
        lines = list(log)

    step, epot, gamma1, gamma2 = _parse_ta_log(lines)

    plt.figure()
    plt.plot(step, epot, linewidth=1)
    plt.xlabel("Monte-Carlo step")
    plt.ylabel("Potential energy  $E_{\\mathrm{pot}}$  (reduced units)")
    plt.title("Potential energy vs. MC step")
    plt.tight_layout()

    plt.figure()
    plt.plot(step, gamma1, label=r"$\\gamma_\\uparrow$ (area increase)")
    plt.plot(step, gamma2, label=r"$\\gamma_\\downarrow$ (area decrease)", linestyle="--")
    plt.xlabel("Monte-Carlo step")
    plt.ylabel(r"Surface tension $\\gamma$ (reduced)")
    plt.title("Surface tension estimates vs. MC step")
    plt.legend()
    plt.tight_layout()
    plt.show()
