import numpy as np

""" 
defining variables outside of loop to save computational power as these will not change
"""
u6 = 2.5**6
u12 = u6**2
u = 4 * ((1.0 / u12) - (1.0 / u6))


class Molecule:
    def __init__(self, position):
        self._q = np.array(position)

    def kinetic_energy(self):
        return 0.5 * np.dot(self.v, self.v)

    def potential_energy(self, other, size):
        """
        potential energy for molecule and each neighbor given formula from task
        """
        delta = self._q - other._q

        delta -= size * np.round(delta / size)

        r2 = np.dot(delta, delta)

        if r2 >= 6.25:
            return 0.0
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        return 4 * (inv_r12 - inv_r6) - u
