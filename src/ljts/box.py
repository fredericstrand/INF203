import numpy as np
from src.ljts.molecule import Molecule
from typing import Optional


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

    def total_potential_energy(self):
        """
        Calculate the total potential energy of all molecules in the system.
        
        Computes the potential energy by summing pairwise interactions
        between all molecules in the box using the potential object.
        """
        Epot = 0.0
        N = len(self._molecules)

        for i in range(N):
            for j in range(i + 1, N):
                pos_i = self._molecules[i].position
                pos_j = self._molecules[j].position

                Epot += self.potential.potential_energy(pos_i, pos_j, self._box_size)

        self._total_Epot = Epot

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
