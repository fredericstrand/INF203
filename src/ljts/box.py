import numpy as np
from src.ljts.molecule import Molecule
from collections import defaultdict
from typing import Optional
import itertools


class Box:
    """
    A class used to represent a simulation box containing molecules.
    
    The Box class manages a collection of molecules within a 3D rectangular
    container and handles potential energy calculations using cell lists for
    efficient neighbor searching.
    
    Attributes
    ----------
    _box_size : numpy.ndarray
        The dimensions of the box as [len_x, len_y, len_z]
    _molecules : list
        List of Molecule objects contained in the box
    potential : object
        Potential energy calculation object
    _total_Epot : float
        Total potential energy of the system
    
    Methods
    -------
    add_molecule(mol)
        Adds a single molecule to the box
    populate_box(den_liq, den_vap)
        Populates the box with molecules using specified liquid and vapor densities
    write_XYZ(path, mode="w")
        Writes molecule positions to an XYZ file format
    total_potential_energy()
        Calculates the total potential energy using 3D cell lists
    """
    
    def __init__(self, len_x, len_y, len_z, 
                 den_liq: Optional[float] = None, 
                 den_vap: Optional[float] = None, 
                 potential = None) -> None:
        """
        Initialize the Box with specified dimensions and optional population.
        
        Parameters
        ------
        len_x : float
            Length of the box in the x-direction
        len_y : float
            Length of the box in the y-direction
        len_z : float
            Length of the box in the z-direction
        den_liq : float, optional
            Liquid phase density for initial population (default is None)
        den_vap : float, optional
            Vapor phase density for initial population (default is None)
        potential : object, optional
            Potential energy calculation object (default is None)
        """
        self._box_size = np.array([abs(len_x), abs(len_y), abs(len_z)])
        self._molecules = []
        self.potential = potential

        if den_liq is not None and den_vap is not None:
            self.populate_box(den_liq, den_vap)

        self._total_Epot = 0.0
        self.total_potential_energy()

    def add_molecule(self, mol: Molecule) -> None:
        """
        Add a single molecule to the box/system.
        
        Parameters
        ----------
        mol : Molecule
            The molecule object to be added to the box
        """
        self._molecules.append(mol)

    def populate_box(self, den_liq: float, den_vap: float) -> None:
        """
        Populate the box with molecules using specified densities.
        
        Creates a distribution of molecules within different zones of the box
        based on liquid and vapor densities. The box is divided into three zones
        with ratios [2:1:2] in the y-direction, where the middle zone has liquid
        density and the outer zones have vapor density.
        
        Note: List comprehension is used for better readability, though map()
        would likely be more efficient for large numbers of molecules.
        
        Parameters
        ----------
        den_liq : float
            Density of molecules in the liquid phase zone
        den_vap : float
            Density of molecules in the vapor phase zones
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

    def write_XYZ(self, path: str, mode: str = "w") -> None:
        """
        Write molecule positions to an XYZ file format.
        
        Parameters
        ----------
        path : str
            File path where the XYZ file will be written
        mode : str
            File opening mode (default is "w" for write)
        """
        with open(path, mode) as file:
            file.write(f"{len(self._molecules)}\n")
            file.write(f"#\n")
            for mol in self._molecules:
                file.write(
                    f"C {mol._position[0]} {mol._position[1]} {mol._position[2]}\n"
                )

    def total_potential_energy(self) -> None:
        """
        Compute total potential energy using a 3D cell list algorithm.
        
        Uses cell lists to efficiently calculate pairwise interactions between
        molecules while avoiding redundant calculations through neighbor searching.
        The total energy is divided by 2 since each pair is calculated twice.
        
        Raises
        ------
        ValueError
            If no potential method has been defined for the box
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

        # Divide by 2 since each pair is calculated twice
        self._total_Epot = 0.5 * Epot

    def _build_cell_list(self, num_cells: np.ndarray, cell_size: float) -> dict:
        """
        Assign molecules to their respective cells in 3D space.
        
        Creates a cell list data structure where molecules are grouped by
        their spatial location within the box for efficient neighbor searching.
        
        Parameters
        ----------
        num_cells : numpy.ndarray
            Number of cells in each dimension [nx, ny, nz]
        cell_size : float
            Size of each cell (equal to the potential cutoff distance)
            
        Returns
        -------
        dict
            Dictionary mapping cell indices to lists of molecules in that cell
        """
        cell_list = defaultdict(list)
        for mol in self._molecules:
            pos = mol.position
            cell_idx = tuple((pos / cell_size).astype(int) % num_cells)
            cell_list[cell_idx].append(mol)
        return cell_list

    def _get_neighbor_cells(self, cell_idx: tuple, num_cells: np.ndarray) -> list:
        """
        Get all neighbor cell indices including the cell itself.
        
        Uses itertools to generate all 27 neighboring cells (including the
        central cell) in a 3D grid. Handles periodic boundary conditions
        by using modulo arithmetic.
        
        Parameters
        ----------
        cell_idx : tuple
            Index of the current cell as (i, j, k)
        num_cells : numpy.ndarray
            Number of cells in each dimension [nx, ny, nz]
            
        Returns
        -------
        list
            List of neighbor cell indices as tuples
        """
        neighbors = []

        # Use itertools library to iterate over all the cell's neighbors
        for offset in itertools.product([-1, 0, 1], repeat=3):
            neighbor = tuple((np.array(cell_idx) + offset) % num_cells)
            neighbors.append(neighbor)

        return neighbors

    @property
    def get_molecules(self) -> list:
        """
        Get the list of molecules in the box.
        
        Returns
        -------
        list
            List of Molecule objects contained in the box
        """
        return self._molecules

    @property
    def total_epot(self) -> float:
        """
        Get the total potential energy of the system.
        
        Returns
        -------
        float
            Total potential energy of all molecules in the box
        """
        return self._total_Epot

    @property
    def num_molecules(self) -> int:
        """
        Get the number of molecules in the box.
        
        Returns
        -------
        int
            Number of molecules currently in the box
        """
        return len(self._molecules)

    @property
    def box_size(self) -> np.ndarray:
        """
        Get the dimensions of the box.
        
        Returns
        -------
        numpy.ndarray
            Box dimensions as [len_x, len_y, len_z]
        """
        return self._box_size
