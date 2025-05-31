import numpy as np


class Box:
    def __init__(self, lx, ly, lz):
        self.size = np.array([lx, ly, lz])
        self.molecules = []

    def add_molecule(self, mol):
        self.molecules.append(mol)

    def total_potential_energy(self):
        Epot = 0.0
        n = len(self.molecules)
        for i in range(n):
            for j in range(i + 1, n):
                Epot += self.molecules[i].potential_energy(self.molecules[j], self.size)
        return Epot
