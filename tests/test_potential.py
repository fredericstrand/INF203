from ljts.potential import LJTS
import pytest
import numpy as np


def test_potential_cutoff():
    potential = LJTS(cutoff=2.5)
    pos_i = np.array([2.6, 0.0, 0.0])
    pos_j = np.array([0.0, 0.0, 0.0])
    box_size = [10.0, 10.0, 10.0]
    assert potential.potential_energy(pos_i, pos_j, box_size) == pytest.approx(0.0)


def test_potential_pbc():
    potential = LJTS(cutoff=2.5)
    pos_i = np.array([0.1, 0.1, 0.1])
    pos_j = np.array([9.9, 9.9, 9.9])
    box_size = [10.0, 10.0, 10.0]
    assert potential.potential_energy(pos_i, pos_j, box_size) != 0.0
