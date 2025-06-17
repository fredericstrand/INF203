from abc import ABC, abstractmethod
import numpy as np


class Potential(ABC):
    """
    Abstract base class for molecular potential energy calculations.
    
    This class defines the interface that all potential energy models must
    implement. Subclasses should provide specific implementations for
    different types of intermolecular interactions.
    
    Methods
    -------
    potential_energy(pos_i, pos_j, box_size)
        Abstract method to calculate potential energy between two molecules
    """
    
    @abstractmethod
    def potential_energy(self, pos_i: np.ndarray, pos_j: np.ndarray, box_size: np.ndarray) -> float:
        """
        Calculate the potential energy between two molecules.
        
        Abstract method that must be implemented by subclasses to define
        the specific form of intermolecular potential energy calculation.
        
        Parameters
        ----------
        pos_i : numpy.ndarray
            Position of the first molecule as a 3D vector [x, y, z]
        pos_j : numpy.ndarray
            Position of the second molecule as a 3D vector [x, y, z]
        box_size : numpy.ndarray
            Dimensions of the simulation box as [len_x, len_y, len_z]
            
        Returns
        -------
        float
            Potential energy between the two molecules
        """
        pass

class LJTS(Potential):
    """
    Lennard-Jones Truncated and Shifted (LJTS) potential implementation.
    
    Implements the Lennard-Jones potential with a spherical cutoff and
    energy shifting to ensure continuity at the cutoff distance. The
    potential is truncated beyond the cutoff radius and shifted so that
    the energy is zero at the cutoff.
    
    Attributes
    ----------
    cutoff : float
        Cutoff distance for the potential interaction
    u : float
        Energy shift value to ensure continuity at cutoff
    
    Methods
    -------
    potential_energy(pos_i, pos_j, box_size)
        Calculate LJTS potential energy between two molecules
    """
    def __init__(self, cutoff: float = 2.5) -> None:
        """
        Initialize the LJTS potential with specified cutoff distance.
        
        Calculates the energy shift value required to make the potential
        continuous at the cutoff distance by setting the potential energy
        to zero at r = cutoff.
        
        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance for potential interactions (default is 2.5)
        """
        self.cutoff = cutoff
        self.u = 4 * ((1.0 / cutoff**12) - (1.0 / cutoff**6))

    def potential_energy(self, pos_i: np.ndarray, pos_j: np.ndarray, box_size: np.ndarray) -> float:
        """
        Calculate LJTS potential energy between two molecules.
        
        Computes the Lennard-Jones potential with truncation and shifting.
        Uses the minimum image convention to handle periodic boundary
        conditions. Returns zero if the distance exceeds the cutoff.
        
        The LJTS potential is given by:
        U(r) = 4 * [(1/r)^12 - (1/r)^6] - u     for r <= r_cutoff
        U(r) = 0                                for r > r_cutoff
        
        where u is the shift value ensuring continuity at r_cutoff.
        
        Parameters
        ----------
        pos_i : numpy.ndarray
            Position of the first molecule as a 3D vector [x, y, z]
        pos_j : numpy.ndarray
            Position of the second molecule as a 3D vector [x, y, z]
        box_size : numpy.ndarray
            Dimensions of the simulation box as [len_x, len_y, len_z]
            
        Returns
        -------
        float
            LJTS potential energy between the two molecules
        """
        delta = pos_i - pos_j
        delta -= box_size * np.round(delta / box_size)

        r2 = np.dot(delta, delta)

        if r2 >= self.cutoff**2:
            return 0.0

        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        return 4 * (inv_r12 - inv_r6) - self.u

class Harmonic(Potential):
    """
    Harmonic potential implementation.
    
    Example class demonstrating how to inherit from the abstract Potential
    base class.
    """
    pass

class PotentialFactory:
    """
    Factory class to register and create Potential instances.
    
    Provides a registry system for different potential types and allows
    creation of potential objects by name with optional parameters.
    
    Attributes
    ----------
    _types : dict
        Dictionary mapping potential names to their corresponding classes
        
    Methods
    -------
    register(name, potential_class)
        Register a new potential type with the factory
    __call__(name, **kwargs)
        Create and return a potential instance by name
    """
    def __init__(self):
        """
        Initialize the PotentialFactory.
        
        Creates an empty registry for potential types.
        """
        self._types = {}

    def register(self, name: str, potential_class: type) -> None:
        """
        Register a new potential type with the factory.
        
        Parameters
        ----------
        name : str
            Name to associate with the potential class
        potential_class : type
            Class that inherits from Potential
            
        Raises
        ------
        TypeError
            If potential_class does not inherit from Potential
        """
        if not issubclass(potential_class, Potential):
            raise TypeError(f"{potential_class} must inherit from Potential")

        self._types[name] = potential_class

    def __call__(self, name: str, **kwargs) -> Potential:
        """
        Create and return a potential instance by name.
        
        Parameters
        ----------
        name : str
            Name of the registered potential type
        **kwargs
            Keyword arguments to pass to the potential constructor
            
        Returns
        -------
        Potential
            Instance of the requested potential type
            
        Raises
        ------
        KeyError
            If the specified potential name is not registered
        """
        if name not in self._types:
            raise KeyError(f"Unknown potential type: {name}")

        return self._types[name](**kwargs)

class Harmonic(Potential):
    """
    Harmonic potential implementation.
    
    Example class demonstrating how to inherit from the abstract Potential
    base class.
    """
    pass

class PotentialFactory:
    """
    Factory class to register and create Potential instances.
    
    Provides a registry system for different potential types and allows
    creation of potential objects by name with optional parameters.
    
    Attributes
    ----------
    _types : dict
        Dictionary mapping potential names to their corresponding classes
        
    Methods
    -------
    register(name, potential_class)
        Register a new potential type with the factory
    __call__(name, **kwargs)
        Create and return a potential instance by name
    """
    def __init__(self):
        """
        Initialize the PotentialFactory.
        
        Creates an empty registry for potential types.
        """
        self._types = {}

    def register(self, name: str, potential_class: type) -> None:
        """
        Register a new potential type with the factory.
        
        Parameters
        ----------
        name : str
            Name to associate with the potential class
        potential_class : type
            Class that inherits from Potential
            
        Raises
        ------
        TypeError
            If potential_class does not inherit from Potential
        """
        if not issubclass(potential_class, Potential):
            raise TypeError(f"{potential_class} must inherit from Potential")

        self._types[name] = potential_class

    def __call__(self, name: str, **kwargs) -> Potential:
        """
        Create and return a potential instance by name.
        
        Parameters
        ----------
        name : str
            Name of the registered potential type
        **kwargs
            Keyword arguments to pass to the potential constructor
            
        Returns
        -------
        Potential
            Instance of the requested potential type
            
        Raises
        ------
        KeyError
            If the specified potential name is not registered
        """
        if name not in self._types:
            raise KeyError(f"Unknown potential type: {name}")

        return self._types[name](**kwargs)

class Harmonic(Potential):
    """
    Just example for showing how abstract methods work.
    """
    pass

class PotentialFactory:
    """
    Factory to register and create Potential instances.
    """
    def __init__(self):
        self._types = {}

    def register(self, name: str, potential_class: type):
        if not issubclass(potential_class, Potential):
            raise TypeError(f"{potential_class} must inherit from Potential")
        self._types[name] = potential_class

    def __call__(self, name: str, **kwargs) -> Potential:
        if name not in self._types:
            raise KeyError(f"Unknown potential type: {name}")
        return self._types[name](**kwargs)