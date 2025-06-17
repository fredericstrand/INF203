from abc import ABC, abstractmethod
import json
import numpy as np
import os
from typing import Dict, Any, List


class Simulation(ABC):
    """
    Base class for Monte Carlo simulations.
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
        pass

class MetropolisMC(Simulation):
    def __init__(self, box, T: float, b: float, *, log_energy: bool = True):
        super().__init__(box, log_energy=log_energy)
        self.T = T
        self.b = b

    def step(self) -> float:
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
            # trial move
            mol.move_random(self.b, self.box.box_size)
            # new energy with the trial position
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


class Orchestrator:
    """
    Orchestrator class for running and controlling Monte Carlo simulations.
    Reads configuration from JSON file and manages the simulation execution.
    """
    
    def __init__(self, config_file: str):
        """
        Initialize the orchestrator with configuration from JSON file.
        
        Args:
            config_file: Path to the JSON configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.box = None
        self.simulation = None
        self.current_step = 0
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def setup_box(self, box_class):
        """
        Setup the simulation box based on configuration.
        
        Args:
            box_class: The Box class to instantiate
        """
        setup_config = self.config["setup"]
        box_size = np.array([setup_config["Lx"], setup_config["Ly"], setup_config["Lz"]])
        
        # Create box instance
        self.box = box_class(box_size)
        
        # Setup compartments if specified
        if "compartments" in setup_config:
            self._setup_compartments(setup_config["compartments"])
    
    def _setup_compartments(self, compartments: List[Dict[str, float]]):
        """
        Setup compartments in the box based on configuration.
        
        Args:
            compartments: List of compartment configurations
        """
        # This method would need to be implemented based on your Box class
        # compartment setup functionality
        for i, compartment in enumerate(compartments):
            density = compartment["density"]
            volume_fraction = compartment["volume_fraction"]
            # Add molecules to compartment based on density and volume_fraction
            # Implementation depends on your Box class methods
            print(f"Setting up compartment {i+1}: density={density}, volume_fraction={volume_fraction}")
    
    def setup_simulation(self, simulation_class, **simulation_kwargs):
        """
        Setup the simulation instance.
        
        Args:
            simulation_class: The Simulation class to instantiate
            **simulation_kwargs: Additional arguments for simulation initialization
        """
        if self.box is None:
            raise RuntimeError("Box must be setup before simulation. Call setup_box() first.")
        
        # Get maximum displacement from config
        max_displacement = self.config.get("control_parameters", {}).get("maximum_displacement")
        
        # Create simulation instance
        self.simulation = simulation_class(
            box=self.box,
            b=max_displacement,
            **simulation_kwargs
        )
    
    def run_simulation(self):
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Get configuration parameters
        total_steps = self.config["steps"]["total"]
        console_frequency = self.config["console_output"]["frequency"]
        trajectory_frequency = self.config["trajectory_output"]["frequency"]
        trajectory_file = self.config["trajectory_output"]["file"]
        reset_sampling_steps = self.config["steps"].get("reset_sampling_at", [])
        
        # Write initial configuration if specified
        if "initial" in self.config["configuration_output"]:
            initial_file = self.config["configuration_output"]["initial"]
            self.box.write_XYZ(initial_file, mode="w")
            print(f"Initial configuration written to {initial_file}")
        
        print(f"Starting simulation with {total_steps} steps...")
        print(f"Console output frequency: {console_frequency}")
        print(f"Trajectory output frequency: {trajectory_frequency}")
        
        # Run simulation
        for step in range(1, total_steps + 1):
            self.current_step = step
            
            # Check if we need to reset sampling
            if step in reset_sampling_steps:
                # TODO:  """Has to get implemented... ..."""
                print(f"Resetting sampling at step {step}")
            
            # Perform one simulation step
            acceptance = self.simulation.step()
            
            # Console output
            if step == 1 or step % console_frequency == 0:
                output = f"Step {step}: Acceptance ratio: {acceptance:.3f}"
                if self.simulation.log_energy:
                    output += f", Potential energy: {self.box._total_Epot:.3f}"
                print(output)
            
            # Trajectory output
            if step == 1 or step % trajectory_frequency == 0:
                mode = "w" if step == 1 else "a"
                self.box.write_XYZ(trajectory_file, mode=mode)
        
        # Write final configuration if specified
        if "final" in self.config["configuration_output"]:
            final_file = self.config["configuration_output"]["final"]
            self.box.write_XYZ(final_file, mode="w")
            print(f"Final configuration written to {final_file}")
        
        print("Simulation completed successfully!")
    
    def print_config_summary(self):
        """Print a summary of the loaded configuration."""
        print("================== Configuration Summary ==================")
        print(f"Box size: {self.config['setup']['Lx']} x {self.config['setup']['Ly']} x {self.config['setup']['Lz']}")
        print(f"Total steps: {self.config['steps']['total']}")
        print(f"Console output frequency: {self.config['console_output']['frequency']}")
        print(f"Trajectory output frequency: {self.config['trajectory_output']['frequency']}")
        print(f"Maximum displacement: {self.config['control_parameters']['maximum_displacement']}")
        
        if "compartments" in self.config["setup"]:
            print(f"Number of compartments: {len(self.config['setup']['compartments'])}")
            for i, comp in enumerate(self.config["setup"]["compartments"]):
                print(f"  Compartment {i+1}: density={comp['density']}, volume_fraction={comp['volume_fraction']}")
        
        if self.config["steps"].get("reset_sampling_at"):
            print(f"Reset sampling at steps: {self.config['steps']['reset_sampling_at']}")
        
        print("==========================================================")
