import numpy as np
from src.ljts.molecule import Molecule
from collections import defaultdict
import itertools


class Box:
    def __init__(self, len_x, len_y, len_z, den_liq=None, den_vap=None, potential=None):
        """
        initializing the box class that will contain all the molecules. and will also generate the distrobution
        """
        self._box_size = np.array([abs(len_x), abs(len_y), abs(len_z)])
        self._molecules = []
        self.potential = potential

        if den_liq is not None and den_vap is not None:
            self.populate_box(den_liq, den_vap)

        self._total_Epot = 0.0
        self.total_potential_energy()

    def add_molecule(self, mol):
        """
        method for adding molecules to the box/system
        """
        self._molecules.append(mol)

    def populate_box(self, den_liq, den_vap):
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

    def total_potential_energy(self):
        """
        Compute total potential energy using a 3D cell list.
        """
        if self.potential is None:
            raise ValueError("Not defined a potential method.")
        cutoff = self.potential.cutoff
        box_size = self._box_size
        cell_size = cutoff
        num_cells = np.floor(box_size / cell_size).astype(int)

        # Build the cell list
        cell_list = self._build_cell_list(num_cells, cell_size)

        Epot = 0.0
        for cell_index, particles in cell_list.items():
            neighbor_cells = self._get_neighbor_cells(cell_index, num_cells)

            for _, mol_i in enumerate(particles):
                pos_i = mol_i.position
                for neighbor_cell in neighbor_cells:
                    for mol_j in cell_list.get(neighbor_cell, []):
                        if mol_i is mol_j:
                            continue  # Avoid self-interaction
                        pos_j = mol_j.position

                        Epot += self.potential.potential_energy(pos_i, pos_j, box_size)

        # Dvide by 2 since each pair is calculated twice
        self._total_Epot = 0.5 * Epot

    def _build_cell_list(self, num_cells, cell_size):
        """
        Assign molecules to their respective cells in 3D space.
        """
        cell_list = defaultdict(list)
        for mol in self._molecules:
            pos = mol.position
            cell_idx = tuple((pos / cell_size).astype(int) % num_cells)
            cell_list[cell_idx].append(mol)
        return cell_list

    def _get_neighbor_cells(self, cell_idx, num_cells):
        """
        Returns a list of neighbor cell indices (including the cell itself).
        Handles periodic boundary conditions.
        """
        neighbors = []

        # Use itertools library to iterate over all the cells neighbors
        for offset in itertools.product([-1, 0, 1], repeat=3):
            neighbor = tuple((np.array(cell_idx) + offset) % num_cells)
            neighbors.append(neighbor)
        return neighbors

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

    @property
    def total_Epot(self):
        return self._total_Epot
