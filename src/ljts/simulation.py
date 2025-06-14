from abc import ABC, abstractmethod
import numpy as np

class Simulation(ABC):
    """
    Abstract base class for Monte Carlo simulations with energy logging capabilities.
    """
    
    def __init__(self, box, log_energy: bool = True):
        """
        Initialize simulation with box system and logging configuration.

        Args:
            box (Box): Box object containing molecules and potential energy functions.
            log_energy (bool, optional): Flag to enable potential energy logging. Defaults to True.

        Returns:
            None
        """
        self.box = box
        self.log_energy = log_energy

    @abstractmethod
    def step(self):
        """
        Execute single Monte Carlo step with acceptance ratio calculation.

        Args:
            None

        Returns:
            float: Acceptance ratio for the Monte Carlo step.
        """
        pass

    def run(self, n_steps: int, log_interval: int = 200, xyz_path: str = None):
        """
        Execute simulation for specified steps with periodic logging and trajectory output.

        Args:
            n_steps (int): Total number of Monte Carlo steps to perform.
            log_interval (int, optional): Interval for logging and XYZ output. Defaults to 200.
            xyz_path (str, optional): Path for XYZ trajectory file output. Defaults to None.

        Returns:
            None
        """
        for step in range(1, n_steps + 1):
            acceptance = self.step()
            output = f"Step {step}: Acceptance ratio: {acceptance:.3f}"
            if self.log_energy:
                output += f", Potential energy: {self.box._total_Epot:.3f}"
            if step == 1 or step % log_interval == 0:
                self.box.write_XYZ("data/trajectory.xyz", mode="a")
                print(output)

class MetropolisMC(Simulation):
    def __init__(self, box, T: float, b: float, *, log_energy: bool = True):
        """
        Initialize Metropolis Monte Carlo simulation with temperature and displacement parameters.

        Args:
            box (Box): Box object containing molecules and potential energy functions.
            T (float): Temperature for Metropolis acceptance criterion.
            b (float): Maximum displacement distance for random moves.
            log_energy (bool, optional): Flag to enable potential energy logging. Defaults to True.

        Returns:
            None
        """
        super().__init__(box, log_energy=log_energy)
        self.T = T
        self.b = b

    def step(self) -> float:
        """
        Perform single Metropolis Monte Carlo step with trial moves for all molecules.

        Args:
            None

        Returns:
            float: Acceptance ratio for the Monte Carlo step (accepted moves / total moves).
        """
        accepted = 0
        N = len(self.box._molecules)
        
        for i in range(N):
            idx = np.random.randint(N)
            mol = self.box._molecules[idx]
            
            # Calculate potential energy before moving
            old_E = sum(
                self.box.potential.potential_energy(
                    mol.position, other.position, self.box.box_size
                )
                for other in self.box.get_molecules
                if other is not mol
            )
            
            # Trial move
            mol.move_random(self.b, self.box.box_size)
            
            # New energy with the trial position
            new_E = sum(
                self.box.potential.potential_energy(
                    mol.alt_position, other.position, self.box.box_size
                )
                for other in self.box.get_molecules
                if other is not mol
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
