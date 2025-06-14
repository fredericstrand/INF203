import numpy as np
from src.ljts.molecule import Molecule
from collections import defaultdict
import itertools


class Box:
    """
    A class used to represent a simulation box containing molecules
    
    ...
    
    Attributes
    ----------
    potential : object
        the potential energy calculation object
    get_molecules : list
        list of all molecules in the box
    total_epot : float
        the total potential energy of the system
    num_molecules : int
        the number of molecules in the box
    box_size : numpy.ndarray
        the dimensions of the box [len_x, len_y, len_z]
    
    Methods
    -------
    add_molecule(mol)
        Adds a molecule to the box/system
    write_XYZ(path, mode="w")
        Writes molecular positions to XYZ file format
    total_potential_energy()
        Computes total potential energy using a 3D cell list
    """
    
    def __init__(self, len_x, len_y, len_z, den_liq=None, den_vap=None, potential=None):
        """
        Initialize the box class that will contain all the molecules and generate the distribution.
        
        Parameters
        ----------
        len_x : float
            Length of the box in x dimension
        len_y : float
            Length of the box in y dimension
        len_z : float
            Length of the box in z dimension
        den_liq : float, optional
            Liquid density for population (default is None)
        den_vap : float, optional
            Vapor density for population (default is None)
        potential : object, optional
            Potential energy calculation object (default is None)
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
        Add a molecule to the box/system.
        
        Parameters
        ----------
        mol : Molecule
            The molecule object to add to the system
        """
        self._molecules.append(mol)

    def _populate_box(self, den_liq, den_vap):
        """
        Populate the box/system by creating a distribution of molecules within different zones.
        
        Uses list comprehension for better readability. 'map()' would likely be more efficient.
        
        Parameters
        ----------
        den_liq : float
            Liquid density for molecular distribution
        den_vap : float
            Vapor density for molecular distribution
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
        Write molecular positions to XYZ file format.
        
        Parameters
        ----------
        path : str
            File path for output XYZ file
        mode : str, optional
            File write mode (default is "w")
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
        Compute total potential energy using a 3D cell list.
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
        Assign molecules to their respective cells in 3D space.
        
        Parameters
        ----------
        num_cells : numpy.ndarray
            Number of cells in each dimension
        cell_size : float
            Size of each cell
            
        Returns
        -------
        defaultdict
            Dictionary mapping cell indices to lists of molecules
        """
        cell_list = defaultdict(list)
        for mol in self._molecules:
            pos = mol.position
            cell_idx = tuple((pos / cell_size).astype(int) % num_cells)
            cell_list[cell_idx].append(mol)
        return cell_list

    def _get_neighbor_cells(self, cell_idx, num_cells):
        """
        Get a list of neighbor cell indices including the cell itself.
        
        Handles periodic boundary conditions using itertools library to iterate 
        over all the cell's neighbors.
        
        Parameters
        ----------
        cell_idx : tuple
            Index of the current cell
        num_cells : numpy.ndarray
            Number of cells in each dimension
            
        Returns
        -------
        list
            List of neighbor cell indices (tuples)
        """
        neighbors = []

        for offset in itertools.product([-1, 0, 1], repeat=3):
            neighbor = tuple((np.array(cell_idx) + offset) % num_cells)
            neighbors.append(neighbor)
        return neighbors

    @property
    def get_molecules(self):
        """
        Access the list of all molecules in the box.
        
        Returns
        -------
        list
            List of Molecule objects in the system
        """
        return self._molecules

    @property
    def total_epot(self):
        """
        Access the total potential energy of the system.
        
        Returns
        -------
        float
            Total potential energy
        """
        return self._total_Epot

    @property
    def num_molecules(self) -> int:
        """
        Access the number of molecules in the box.
        
        Returns
        -------
        int
            Number of molecules in the system
        """
        return len(self._molecules)

    @property
    def box_size(self):
        """
        Access the dimensions of the simulation box.
        
        Returns
        -------
        numpy.ndarray
            Box dimensions [len_x, len_y, len_z]
        """
        return self._box_size
