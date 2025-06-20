import numpy as np
from src.ljts.distortion import compute_distortion
from src.ljts.box import Box
from src.ljts.molecule import Molecule
from src.ljts.potential import LJTS


def test_vol_increase():
    zeta = 1.000001
    sx = zeta
    sy = 1 / zeta**2
    sz = zeta
    assert abs(sx * sy * sz - 1.0) < 1e-10, "the volume is not 1, increase"


def test_vol_decrease():
    zeta = 1.000001
    sx = 1 / zeta
    sy = zeta**2
    sz = 1 / zeta
    assert abs(sx * sy * sz - 1.0) < 1e-10, "the volume is not 1, decrease"


def test_delta_u():
    box = Box(
        len_x=5.0,
        len_y=5.0,
        len_z=5.0,
        den_liq=0.73,
        den_vap=0.02,
        potential=LJTS(cutoff=2.5),
    )
    box.populate_box(den_liq=0.73, den_vap=0.02)
    zeta = 1
    delta_u, delta_a = compute_distortion(box, zeta, 1 / zeta**2, zeta)
    assert abs(delta_u) < 1e-10, "fault in distortion, delta u should be 0"


def test_delta_a():
    box = Box(
        len_x=5.0,
        len_y=5.0,
        len_z=5.0,
        den_liq=0.73,
        den_vap=0.02,
        potential=LJTS(cutoff=2.5),
    )
    box.populate_box(den_liq=0.73, den_vap=0.02)
    zeta = 1
    delta_u, delta_a = compute_distortion(box, zeta, 1 / zeta**2, zeta)
    assert abs(delta_a) < 1e-10, "fault in distortion, delta a should be 0"
