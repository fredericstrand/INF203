from abc import ABC, abstractmethod
import numpy as np

class Potential(ABC):
    """
    Abstract base class for potential energy calculations between molecules
    
    ...
    
    Methods
    -------
    potential_energy(pos_i, pos_j, box_size)
        Calculate potential energy between two molecules using minimum image convention
    """
    
    @abstractmethod
    def potential_energy(self, pos_i: np.ndarray, pos_j: np.ndarray, box_size: np.ndarray) -> float:
        """
        Calculate potential energy between two molecules using minimum image convention.
        
        Parameters
        ----------
        pos_i : numpy.ndarray
            Position vector of first molecule
        pos_j : numpy.ndarray
            Position vector of second molecule
        box_size : numpy.ndarray
            Box dimensions for periodic boundary conditions
            
        Returns
        -------
        float
            Potential energy between the two molecules
        """
        pass

class LJTS(Potential):
    """
    Lennard-Jones truncated and shifted potential energy calculator
    
    ...
    
    Attributes
    ----------
    cutoff : float
        the cutoff distance for potential truncation
    u : float
        the shift value for potential energy correction
    
    Methods
    -------
    potential_energy(pos_i, pos_j, box_size)
        Calculate Lennard-Jones potential energy between two molecules with truncation and shifting
    """
    
    def __init__(self, cutoff: float = 2.5):
        """
        Initialize Lennard-Jones truncated and shifted potential with cutoff distance.
        
        Parameters
        ----------
        cutoff : float, optional
            Cutoff distance for potential truncation (default is 2.5)
        """
        self.cutoff = cutoff
        self.u = 4 * ((1.0 / cutoff**12) - (1.0 / cutoff**6))
    
    def potential_energy(self, pos_i, pos_j, box_size):
        """
        Calculate Lennard-Jones potential energy between two molecules with truncation and shifting.
        
        Parameters
        ----------
        pos_i : numpy.ndarray
            Position vector of first molecule
        pos_j : numpy.ndarray
            Position vector of second molecule
        box_size : numpy.ndarray
            Box dimensions for periodic boundary conditions
            
        Returns
        -------
        float
            Lennard-Jones potential energy between molecules, zero beyond cutoff
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
