from abc import ABC, abstractmethod
import numpy as np

class Simulation(ABC):
    """
    Base class for Monete Carlo simulations.

    Subclasses must implement the 'step()' -> float (acceptance ratio) method.
    """

    def __init__(self, box, log_energy: bool = True):
        self.box = box
        self.log_energy = log_energy

    @abstractmethod
    def step(self):
        """
        Perform one Monte Carlo step and return the acceptance ratio.
        """

    def run(self, n_steps: int, log_interval: int = 50):
        """
        Run the simulation for a specified number of steps, logging the potential energy and acceptance ratio at specified intervals.
        """
        for step in range(1, n_steps + 1):
            acceptance = self.step()
            output = f"Step {step}: Acceptance ratio: {acceptance:.3f}"

            if self.log_energy:
                output += f", Potential energy: {self.box._total_Epot:.3f}"
            if step == 1 or s % log_interval == 0:
                print(output)

class MetropolisMC(Simulation):
    def __init__(self, box, T: float, b: float):
        super().__init__(box, log_every = log_every)
        self.T = T
        self.b = b

    def step() -> float:
        accepted = 0
        N = len(self.box._molecules)
        for i in range(N):
            idx = np.random.randint(N)
            mol = self.box._molecules[idx]
            
            # Calculate potential energy before moving
            old_E = sum(
                mol.potential_energy(other, self.box._size, use_alt_self=False)
                for other in self.box._molecules if other is not mol
            )

            # Move the molecule randomly
            mol.move_random(self.b, self.box._size)

            # Calculate potential energy after moving
            new_E = sum(
                mol.potential_energy(other, self.box._size, use_alt_self=True)
                for other in self.box._molecules if other is not mol
            )

            delta_E = new_E - old_E

            # Accept or reject the move based on Metropolis criterion
            if delta_E < 0 or np.random.rand() < np.exp(-delta_E / self.T):
                mol.position = np.copy(mol.alt_position)
                accepted += 1
            else:
                mol.reset_alt_position()

        self.box.total_potential_energy()
        return accepted / N

