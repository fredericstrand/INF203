import numpy as np


class Molecule:
    def __init__(self, position):
        """
        initializing the class  with positional arrays from argument from box class in the method populate box

        it also checks if the position is the correct length (x,y,z) for error handling
        also added data encapsulation

        _position is the "main" position and will get updated if the metroplis algorithm succeeds or the potential energy decreases
        _alt_position is the "temperary" position that will be used in the simulation
        """
        if len(position) != 3:
            raise ValueError("Position must be a 3D vector with three components.")

        self._position = np.array(position)
        self._alt_position = np.array(position)

    def reset_alt_position(self):
        """
        method for resetting the position if it fails and prior value is better
        """
        self._alt_position = np.copy(self._position)

    def move_random(self, b, box_size):
        """
        moves the alternativ position randomly in a uniform way within the distance of b in all dimensions
        """
        self._alt_position = (
            self._alt_position + np.random.uniform(-b, b, 3)
        ) % box_size

    """ 
    defined some getters and setters
    
    getters for position and alternativ position
    
    a setter for position for modifying the position from outside of the class.
    """

    @property
    def position(self):
        return self._position

    @property
    def alt_position(self):
        return self._alt_position

    @position.setter
    def position(self, new_position):
        if len(new_position) != 3:
            raise ValueError("New position must be a 3D vector with three components.")
        self._position = np.array(new_position)
