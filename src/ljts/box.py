import numpy as np
from src.ljts.molecule import Molecule


class Box:
    def __init__(self, len_x, len_y, len_z, den_liq=None, den_vap=None):
        """
        initializing the box class that will contain all the molecules. and will also generate the distrobution
        """
        self._size = np.array([len_x, len_y, len_z])
        self._molecules = []
        self._total_Epot = 0

        if den_liq and den_vap:
            self._populate_box(den_liq, den_vap)

    def add_molecule(self, mol):
        """
        method for adding molecules to the box/system
        """
        self._molecules.append(mol)

    def total_potential_energy(self):
        """
        calculating the total potential energy of all molecules withing the box/system by calculating the potential energy between molecule and the neighbors.
        """
        Epot = 0.0
        n = len(self._molecules)
        for i in range(n):
            for j in range(i + 1, n):
                Epot += self._molecules[i].potential_energy(
                    self._molecules[j], self._size
                )
        self._total_Epot = Epot

    def _populate_box(self, den_liq, den_vap):
        """
        function for populating the box/system by taking the parameters defined in main script and from that creating a distrobution of molecules within the different zones
        """
        len_x, len_y, len_z = self._size
        vol = len_x * len_y * len_z

        ratios = np.array([2, 1, 2])
        fractions = ratios / ratios.sum()
        boundaries = np.cumsum(np.insert(fractions * len_y, 0, 0))
        zone = fractions * vol
        densities = [den_vap, den_liq, den_vap]
        mol_count = (densities * zone).astype(int)

        for (max, min), count in zip(zip(boundaries[:-1], boundaries[1:]), mol_count):
            pos = np.random.uniform(
                low=(0, min, 0), high=(len_x, max, len_z), size=(count, 3)
            )
            self._molecules.extend([Molecule(pos) for pos in pos])

    def simulation(self, T, b):
        accepted = 0

        N = len(self._molecules)
        for i in range(N):
            id = np.random.randint(N)
            mol = self._molecules[id]
            old_E = 0.0
            new_E = 0.0

            mol.move_random(b, self._size)

            for other in self._molecules:
                if other is mol:
                    continue

                old_E += mol.potential_energy(other, self._size, use_alt_self=False)
                new_E += mol.potential_energy(other, self._size, use_alt_self=True)
            delta_E = new_E - old_E

            if delta_E <= 0.0 or np.random.rand() < np.exp(-delta_E / T):
                mol._position = np.copy(mol._alt_position)
                accepted += 1
        self.total_potential_energy()
        return accepted / N

    """
    defining the getters, no need for setters
    """

    @property
    def get_molecules(self):
        return self._molecules

    @property
    def get_total_epot(self):
        return self._total_Epot
