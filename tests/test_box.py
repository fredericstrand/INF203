from src.ljts.box import Box
import pytest
import numpy as np


def test_box_size():
    box = Box(10.0, 10.0, 10.0)
    assert np.allclose(
        box.size, [10.0, 10.0, 10.0]
    ), "the length in each dimension does not match "


def test_volume():
    box = Box(10.0, 10.0, 10.0)
    assert box.volume == pytest.approx(
        24.0
    ), "the volume of the box does not match given the dimensions"


def test_populate():
    box = Box(10.0, 10.0, 10.0)
    box.populate_box(10)
    assert (
        len(box._molecules) == 10
    ), "input num molecules does not match the real num molecules"


def test_inside_box():
    box = Box(10.0, 10.0, 10.0)
    box.populate_box(10)
    for mol in box._molecules:
        mol.move_random(0.5, 8.0)
    assert all(
        np.all((mol.alt_position >= 0) & (mol.alt_position < 8.0))
        for mol in box._molecules
    ), "at least one molecule is outside of the box"
