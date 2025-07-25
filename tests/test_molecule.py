from src.ljts.molecule import Molecule
import numpy as np
import pytest


def test_position():
    mol = Molecule(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(
        mol.position, [1.0, 2.0, 3.0]
    ), "position given to constructor is not molecules position"


@pytest.mark.parametrize(
    "invalid_position", [[1.0, 2.0], [1.0], [], [1.0, 2.0, 3.0, 4.0]]
)
def test_position_dimension(invalid_position):
    with pytest.raises(
        ValueError, match="Position must be a 3D vector with exactly three components."
    ):
        Molecule(invalid_position)


def test_move_random():
    mol = Molecule(np.array([3.0, 2.0, 1.0]))
    box_size = np.array([1, 2, 5])
    b = 0.5
    mol.move_random(b, box_size)
    delta = (mol.alt_position - mol.position + box_size / 2) % box_size - box_size / 2
    assert np.all(np.abs(delta) <= b), "the molecule is moved outside of the box"


def test_getter_position():
    mol = Molecule(np.array([4.0, 5.0, 6.0]))
    assert np.allclose(mol.position, [4.0, 5.0, 6.0])


def test_getter_alt_position():
    mol = Molecule(np.array([7.0, 8.0, 9.0]))
    assert np.allclose(mol.alt_position, [7.0, 8.0, 9.0])


def test_setter_position():
    mol = Molecule(np.array([0.0, 0.0, 0.0]))
    mol.position = np.array([7.0, 8.0, 9.0])
    assert np.allclose(mol.position, [7.0, 8.0, 9.0])
