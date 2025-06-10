import numpy as np
from src.ljts.molecule import Molecule


class Box:
    def __init__(self, len_x, len_y, len_z, den_liq=None, den_vap=None, potential=None):
        """
        initializing the box class that will contain all the molecules. and will also generate the distrobution
        """
        self._box_size = np.array([len_x, len_y, len_z])
        self._molecules = []
        self.potential = potential

        if den_liq is not None and den_vap is not None:
            self._populate_box(den_liq, den_vap)

        self._total_Epot = 0.0
        self.total_potential_energy()

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
        N = len(self._molecules)

        for i in range(N):
            for j in range(i + 1, N):
                pos_i = self._molecules[i].position
                pos_j = self._molecules[j].position

                Epot += self.potential.potential_energy(pos_i, pos_j, self._box_size)

        self._total_Epot = Epot

    def _populate_box(self, den_liq, den_vap):
        """
        function for populating the box/system by taking the parameters defined in main script and from that creating a distrobution of molecules within the different zones

        We used 'map()' to add the moleulces in worksheet 1, but opted for a list comprehension for better readability. 'map()' would likely be more efficient.
        """
        len_x, len_y, len_z = self._box_size
        vol = len_x * len_y * len_z

        ratios = np.array([2, 1, 2])
        fractions = ratios / ratios.sum()
        boundaries = np.cumsum(np.insert(fractions * len_y, 0, 0))
        zone = fractions * vol
        densities = [den_vap, den_liq, den_vap]
        mol_count = (densities * zone).astype(int)

        for (min, max), count in zip(zip(boundaries[:-1], boundaries[1:]), mol_count):
            pos = np.random.uniform(
                low=(0, min, 0), high=(len_x, max, len_z), size=(count, 3)
            )
            self._molecules.extend([Molecule(pos) for pos in pos])

    def write_XYZ(self, path, mode="w"):
        with open(path, mode) as file:
            file.write(f"{len(self._molecules)}\n")
            file.write(f"#\n")
            for mol in self._molecules:
                file.write(
                    f"C {mol._position[0]} {mol._position[1]} {mol._position[2]}\n"
                )

    @property
    def get_molecules(self):
        return self._molecules

    @property
    def total_epot(self):
        return self._total_Epot

    @property
    def num_molecules(self) -> int:
        return len(self._molecules)

    @property
    def box_size(self):
        return self._box_size
