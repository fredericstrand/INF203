import numpy as np

class Molecule:
    def __init__(self, position):
        """
        Initialize molecule with position coordinates and alternative position for simulation.

        Args:
            position (array-like): 3D position vector [x, y, z] coordinates.

        Returns:
            None

        Raises:
            ValueError: If position is not a 3D vector with three components.
        """
        if len(position) != 3:
            raise ValueError("Position must be a 3D vector with three components.")
        self._position = np.array(position)
        self._alt_position = np.array(position)

    def reset_alt_position(self):
        """
        Reset alternative position to current main position.

        Args:
            None

        Returns:
            None
        """
        self._alt_position = np.copy(self._position)

    def move_random(self, b, box_size):
        """
        Apply random displacement to alternative position with periodic boundary conditions.

        Args:
            b (float): Maximum displacement magnitude in each dimension.
            box_size (numpy.ndarray): Box dimensions for periodic boundary wrapping.

        Returns:
            None
        """
        self._alt_position = (
            self._alt_position + np.random.uniform(-b, b, 3)
        ) % box_size

    @property
    def position(self):
        """
        Access the main position coordinates.

        Args:
            None

        Returns:
            numpy.ndarray: Current position vector [x, y, z].
        """
        return self._position

    @property
    def alt_position(self):
        """
        Access the alternative position coordinates used in simulation trials.

        Args:
            None

        Returns:
            numpy.ndarray: Alternative position vector [x, y, z].
        """
        return self._alt_position

    @position.setter
    def position(self, new_position):
        """
        Set new main position coordinates with validation.

        Args:
            new_position (array-like): New 3D position vector [x, y, z] coordinates.

        Returns:
            None

        Raises:
            ValueError: If new_position is not a 3D vector with three components.
        """
        if len(new_position) != 3:
            raise ValueError("New position must be a 3D vector with three components.")
        self._position = np.array(new_position)
