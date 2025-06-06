import numpy as np

class Molecule:
    def __init__(self, position):
        if len(position) != 3:
            raise ValueError("Position must be a 3D vector with three components.")

        self._x = position[0]
        self._y = position[1]
        self._z = position[2]
        self._position = np.array(position)
        self._u = 4 * ((1.0 / 2.5**12) - (1.0 / 2.5**6))

    def kinetic_energy(self):
        return 0.5 * np.dot(self.v, self.v)

    def potential_energy(self, other, size):
        """
        potential energy for molecule and each neighbor given formula from task
        """
        delta = self._position - other._position

        delta -= size * np.round(delta / size)

        r2 = np.dot(delta, delta)

        if r2 >= 6.25:
            return 0.0

        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        return 4 * (inv_r12 - inv_r6) - self._u
