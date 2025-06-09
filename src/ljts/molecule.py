import numpy as np

class Molecule:
    def __init__(self, position):
        if len(position) != 3:
            raise ValueError("Position must be a 3D vector with three components.")

        self._position = np.array(position)
        self._alt_position = np.array(position)

    def reset_alt_position(self):
        self._alt_position = np.copy(self._position)

    def move_random(self, b, box_size):
        self._alt_position = (self._alt_position + np.random.uniform(-b, b, 3)) % box_size
