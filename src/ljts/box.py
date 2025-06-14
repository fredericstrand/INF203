import numpy as np
from src.ljts.molecule import Molecule
from collections import defaultdict
import itertools


class Box:
    def __init__(self, len_x, len_y, len_z, den_liq=None, den_vap=None, potential=None):
        """
        Initialize the box container for molecules and generate initial distribution.

        Args:
            len_x (float): Box dimension in x-direction.
            len_y (float): Box dimension in y-direction.
            len_z (float): Box dimension in z-direction.
            den_liq (float, optional): Liquid density for population. Defaults to None.
            den_vap (float, optional): Vapor density for population. Defaults to None.
            potential (Potential, optional): Potential energy function object. Defaults to None.

        Returns:
            None
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
        Add a single molecule to the box system.

        Args:
            mol (Molecule): Molecule object to be added to the system.

        Returns:
            None
        """
        self._molecules.append(mol)

    def _populate_box(self, den_liq, den_vap):
        """
        Populate the box with molecules distributed across vapor-liquid zones.

        Args:
            den_liq (float): Liquid phase density for molecule distribution.
            den_vap (float): Vapor phase density for molecule distribution.

        Returns:
            None
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
        """
        Write molecular positions to XYZ format file.

        Args:
            path (str): File path for output XYZ file.
            mode (str, optional): File write mode. Defaults to "w".

        Returns:
            None
        """
        with open(path, mode) as file:
            file.write(f"{len(self._molecules)}\n")
            file.write(f"#\n")
            for mol in self._molecules:
                file.write(
                    f"C {mol._position[0]} {mol._position[1]} {mol._position[2]}\n"
                )

    def total_potential_energy(self):
        """
        Compute total potential energy using 3D cell list optimization.

        Args:
            None

        Returns:
            None
        """
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

        # Divide by 2 since each pair is calculated twice
        self._total_Epot = 0.5 * Epot

    def _build_cell_list(self, num_cells, cell_size):
        """
        Assign molecules to their respective cells in 3D space for neighbor search.

        Args:
            num_cells (numpy.ndarray): Number of cells in each dimension.
            cell_size (float): Size of each cell for spatial partitioning.

        Returns:
            defaultdict: Dictionary mapping cell indices to lists of molecules.
        """
        cell_list = defaultdict(list)
        for mol in self._molecules:
            pos = mol.position
            cell_idx = tuple((pos / cell_size).astype(int) % num_cells)
            cell_list[cell_idx].append(mol)
        return cell_list

    def _get_neighbor_cells(self, cell_idx, num_cells):
        """
        Generate neighbor cell indices including periodic boundary conditions.

        Args:
            cell_idx (tuple): Current cell index coordinates.
            num_cells (numpy.ndarray): Total number of cells in each dimension.

        Returns:
            list: List of neighboring cell indices including the current cell.
        """
        neighbors = []

        # Use itertools library to iterate over all the cells neighbors
        for offset in itertools.product([-1, 0, 1], repeat=3):
            neighbor = tuple((np.array(cell_idx) + offset) % num_cells)
            neighbors.append(neighbor)
        return neighbors

    @property
    def get_molecules(self):
        """
        Access the list of molecules in the box.

        Args:
            None

        Returns:
            list: List of Molecule objects in the system.
        """
        return self._molecules

    @property
    def total_epot(self):
        """
        Access the total potential energy of the system.

        Args:
            None

        Returns:
            float: Total potential energy value.
        """
        return self._total_Epot

    @property
    def num_molecules(self) -> int:
        """
        Get the number of molecules in the box.

        Args:
            None

        Returns:
            int: Total count of molecules in the system.
        """
        return len(self._molecules)

    @property
    def box_size(self):
        """
        Access the box dimensions.

        Args:
            None

        Returns:
            numpy.ndarray: Array containing box dimensions [len_x, len_y, len_z].
        """
        return self._box_size
