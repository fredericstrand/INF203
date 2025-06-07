import numpy as np


class Molecule:
    def __init__(self, position):
        if len(position) != 3:
            raise ValueError("Position must be a 3D vector with three components.")

        self._position = np.array(position)
        self._alt_position = np.array(position)
        self._u = 4 * ((1.0 / 2.5**12) - (1.0 / 2.5**6))

    """ def kinetic_energy(self):
        return 0.5 * np.dot(self.v, self.v) """

    def reset_alt_position(self):
        self._alt_position = np.copy(self._position)

    def move_random(self, b, box_size):
        self._alt_position = (
            self._alt_position + np.random.uniform(-b, b, 3)
        ) % box_size

    def potential_energy(self, other, size, use_alt_self=False):
        """
        potential energy for molecule and each neighbor given formula from task
        """
        pos_i = self._alt_position if use_alt_self else self._position

        delta = pos_i - other._position

        delta -= size * np.round(delta / size)

        r2 = np.dot(delta, delta)

        if r2 >= 6.25:
            return 0.0

        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        return 4 * (inv_r12 - inv_r6) - self._u

    """ @alt_position.setter
    def alt_position(self, value):
        self._alt_position = np.copy(value)

    @property
    def position(self):
        return self._position """
