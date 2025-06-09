from abs import ABC, abstractmethod
import numpy as np

class Potential(ABC):
    @abstractmethod
    def potential_energy(self, other, box_size) -> float:
        """
        Abstract method to calculate the potential energy between this molecule and another.
        Subclasses must implement this method.
        """
        pass

class LJTS(Potential):
    def __init__(self, cutoff: float = 2.5):
        self.cutoff = cutoff
        self.u = 4 * ((1.0 / cutoff**12) - (1.0 / cutoff**6))

    def potential_energy(self, other, box_size):
        """
        Calculate the potential energy between this molecule and another using the Lennard-Jones potential.
        """
        delta = self._position - other._position
        delta -= box_size * np.round(delta / box_size)

        r2 = np.dot(delta, delta)

        if r2 >= self.cutoff**2:
            return 0.0

        inv_r2 = 1.0 / r2
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        return 4 * (inv_r12 - inv_r6) - self.u