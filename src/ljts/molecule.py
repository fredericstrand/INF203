import numpy as np

class Molecule:
    """
    A class used to represent a Molecule with position tracking for simulations
    
    ...
    
    Attributes
    ----------
    position : numpy.ndarray
        the main 3D position vector [x, y, z] coordinates
    alt_position : numpy.ndarray
        the alternative 3D position vector [x, y, z] used in simulation trials
    
    Methods
    -------
    reset_alt_position()
        Resets alternative position to current main position
    move_random(b, box_size)
        Applies random displacement to alternative position with periodic boundary conditions
    """
    
    def __init__(self, position):
        """
        Initialize molecule with position coordinates and alternative position for simulation.
        
        Parameters
        ----------
        position : array-like
            3D position vector [x, y, z] coordinates
            
        Raises
        ------
        ValueError
            If position is not a 3D vector with three components
        """
        if len(position) != 3:
            raise ValueError("Position must be a 3D vector with three components.")
        self._position = np.array(position)
        self._alt_position = np.array(position)
    
    def reset_alt_position(self):
        """
        Reset alternative position to current main position.
        """
        self._alt_position = np.copy(self._position)
    
    def move_random(self, b, box_size):
        """
        Apply random displacement to alternative position with periodic boundary conditions.
        
        Parameters
        ----------
        b : float
            Maximum displacement magnitude in each dimension
        box_size : numpy.ndarray
            Box dimensions for periodic boundary wrapping
        """
        self._alt_position = (
            self._alt_position + np.random.uniform(-b, b, 3)
        ) % box_size
    
    @property
    def position(self):
        """
        Access the main position coordinates.
        
        Returns
        -------
        numpy.ndarray
            Current position vector [x, y, z]
        """
        return self._position
    
    @property
    def alt_position(self):
        """
        Access the alternative position coordinates used in simulation trials.
        
        Returns
        -------
        numpy.ndarray
            Alternative position vector [x, y, z]
        """
        return self._alt_position
    
    @position.setter
    def position(self, new_position):
        """
        Set new main position coordinates with validation.
        
        Parameters
        ----------
        new_position : array-like
            New 3D position vector [x, y, z] coordinates
            
        Raises
        ------
        ValueError
            If new_position is not a 3D vector with three components
        """
        if len(new_position) != 3:
            raise ValueError("New position must be a 3D vector with three components.")
        self._position = np.array(new_position)
