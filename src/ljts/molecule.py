import numpy as np


class Molecule:
    """
    A class used to represent a molecule in a molecular simulation.
    
    The Molecule class handles position management for molecular dynamics
    simulations, including a main position and an alternative position for
    trial moves in Monte Carlo algorithms like Metropolis sampling.
    
    Attributes
    ----------
    _position : numpy.ndarray
        The current accepted position of the molecule as a 3D vector
    _alt_position : numpy.ndarray
        The alternative/trial position used during simulation moves
    
    Methods
    -------
    reset_alt_position()
        Resets the alternative position to match the current position
    move_random(b, box_size)
        Performs a random displacement move within specified bounds
    """
    
    def __init__(self, position: np.ndarray) -> None:
        """
        Initialize the Molecule with a 3D position.
        
        Sets up both the main position and alternative position with the same
        initial coordinates. The main position represents the accepted state,
        while the alternative position is used for trial moves in simulations.
        
        Parameters
        ----------
        position : np.ndarray
            Initial 3D position as [x, y, z] coordinates
            
        Raises
        ------
        ValueError
            If position is not a 3D vector with exactly three components
        """

        if len(position) != 3:
            raise ValueError("Position must be a 3D vector with exactly three components.")

        self._position = position.copy()
        self._alt_position = position.copy()
    
    def reset_alt_position(self) -> None:
        """
        Reset the alternative position to match the current position.
        
        This method is typically called when a trial move is rejected
        and the alternative position needs to be restored to the last
        accepted configuration.
        """
        self._alt_position = np.copy(self._position)
    
    def move_random(self, b: float, box_size: np.ndarray) -> None:
        """
        Perform a random displacement move on the alternative position.
        
        Generates a uniform random displacement within the range [-b, b]
        in all three dimensions and applies periodic boundary conditions
        to keep the molecule within the simulation box.
        
        Parameters
        ----------
        b : float
            Maximum displacement distance in each dimension
        box_size : numpy.ndarray
            Dimensions of the simulation box as [len_x, len_y, len_z]
        """
        self._alt_position = (
            self._alt_position + np.random.uniform(-b, b, 3)
        ) % box_size
    
    @property
    def position(self) -> np.ndarray:
        """
        Get the current accepted position of the molecule.
        
        Returns
        -------
        numpy.ndarray
            Current position as a 3D vector [x, y, z]
        """
        return self._position
    
    @property
    def alt_position(self) -> np.ndarray:
        """
        Get the alternative/trial position of the molecule.
        
        Returns
        -------
        numpy.ndarray
            Alternative position as a 3D vector [x, y, z]
        """
        return self._alt_position
    
    @position.setter
    def position(self, new_position: np.ndarray) -> None:
        """
        Set a new position for the molecule.
        
        Updates the main position of the molecule. This is typically used
        to accept a trial move by setting the position to the alternative
        position coordinates.
        
        Parameters
        ----------
        new_position : numpy.ndarray
            New 3D position as [x, y, z] coordinates
            
        Raises
        ------
        ValueError
            If new_position is not a 3D vector with exactly three components
        """
        if len(new_position) != 3:
            raise ValueError("New position must be a 3D vector with three components.")
        self._position = np.array(new_position)
