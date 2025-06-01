import numpy as np
from src.ljts.molecule import Molecule


class Box:
    def __init__(self, lx, ly, lz):
        """
        initializing the box class that will contain all the molecules. and will also generate the distrobution
        """
        self.size = np.array([lx, ly, lz])
        self.molecules = []

    def add_molecule(self, mol):
        """
        method for adding molecules to the box/system
        """
        self.molecules.append(mol)

    def total_potential_energy(self):
        """
        calculating the total potential energy of all molecules withing the box/system by calculating the potential energy between molecule and the neighbors.
        """
        Epot = 0.0
        n = len(self.molecules)
        for i in range(n):
            for j in range(i + 1, n):
                Epot += self.molecules[i].potential_energy(self.molecules[j], self.size)
        return Epot

    def populate_box(self, den_liq, den_vap):
        """
        function for populating the box/system by taking the parameters defined in main script and from that creating a distrobution of molecules within the different zones
        """
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
