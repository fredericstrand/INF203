from src.ljts.box import Box
from src.ljts.potential import LJTS
import pytest
import numpy as np


def test_box_size():
    box = Box(10.0, 10.0, 10.0, potential=LJTS())
    assert np.allclose(
        box.box_size, [10.0, 10.0, 10.0]
    ), "the length in each dimension does not match "


def test_volume():
    box = Box(10.0, 10.0, 10.0, potential=LJTS())
    assert np.prod(box.box_size) == pytest.approx(
        1000.0
    ), "the volume of the box does not match given the dimensions"


def test_populate():
    box = Box(5.0, 40.0, 5.0, potential=LJTS())
    box.populate_box(den_liq=0.73, den_vap=0.02)
    assert (
        len(box._molecules) == 162
    ), "input num molecules does not match the real num molecules"


def test_inside_box():
    box = Box(10.0, 10.0, 10.0, potential=LJTS())
    box.populate_box(den_liq=0.73, den_vap=0.02)
    for mol in box._molecules:
        mol.move_random(0.5, 8.0)
    assert all(
        np.all((mol.alt_position >= 0) & (mol.alt_position < 8.0))
        for mol in box._molecules
    ), "at least one molecule is outside of the box"
