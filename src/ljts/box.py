import numpy as np
from src.ljts.molecule import Molecule


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

    def populate_box(self, den_liq, den_vap):
        lx, ly, lz = self.size
        vol = lx * ly * lz

        ratios = np.array([2, 1, 2])
        fractions = ratios / ratios.sum()
        boundaries = np.cumsum(np.insert(fractions * ly, 0, 0))
        zone = fractions * vol
        densities = [den_vap, den_liq, den_vap]
        mol_count = (densities * zone).astype(int)

        for (max, min), count in zip(zip(boundaries[:-1], boundaries[1:]), mol_count):
            pos = np.random.uniform(
                low=(0, min, 0), high=(lx, max, lz), size=(count, 3)
            )
            self.molecules.extend(map(Molecule, pos))
