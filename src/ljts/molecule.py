import numpy as np

u6 = 2.5**6
u12 = u6**2
cutoff = (4 * ((1.0 / u12) - (1.0 / u6))) ** 2


class Molecule:
    def __init__(self, id, x, y, z, vx=0.0, vy=0.0, vz=0.0):
        self.id = id
        self.q = np.array([x, y, z])
        self.v = np.array([vx, vy, vz])

    def kinetic_energy(self):
        return 0.5 * np.dot(self.v, self.v)

    def potential_energy(self, other, size):
        delta = self.q - other.q

        delta -= self.size * np.round(delta / size)

        r2 = np.dot(delta, delta)

        if r2 >= cutoff:
            return 0.0
        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        return 4 * (inv_r12 - inv_r6) - cutoff
