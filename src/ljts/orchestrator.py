from abc import ABC, abstractmethod
import json
import numpy as np
import os
from typing import Any, Dict, List, Optional

from src.ljts.box import Box


class Simulation(ABC):
    """
    Abstract base class for Monte Carlo simulations.
    
    This class defines the interface that all Monte Carlo simulation
    implementations must follow. Subclasses should implement specific
    Monte Carlo algorithms such as Metropolis sampling.
    
    Attributes
    ----------
    box : Box
        The simulation box containing molecules
    log_energy : bool
        Whether to track and log potential energy during simulation
    
    Methods
    -------
    step()
        Abstract method to perform one Monte Carlo step
    """

    def __init__(self, box: Box, log_energy: bool = True) -> None:
        """
        Initialize the base simulation class.
        
        Parameters
        ----------
        box : Box
            The simulation box containing molecules and potential
        log_energy : bool, optional
            Whether to track potential energy during simulation (default is True)
        """
        self.box = box
        self.log_energy = log_energy

    @abstractmethod
    def step(self):
        """
        Perform one Monte Carlo step and return the acceptance ratio.
        
        Abstract method that must be implemented by subclasses to define
        the specific Monte Carlo algorithm used for sampling.
        
        Returns
        -------
        float
            Acceptance ratio for the Monte Carlo step
        """
        pass

class MetropolisMC(Simulation):
    """
    Metropolis Monte Carlo simulation implementation.
    
    Implements the Metropolis algorithm for molecular Monte Carlo simulations.
    Performs random displacement moves on molecules and accepts or rejects
    them based on the Metropolis criterion using the Boltzmann factor.
    
    Attributes
    ----------
    T : float
        Temperature of the simulation (in reduced units)
    b : float
        Maximum displacement parameter for random moves
    
    Methods
    -------
    step()
        Perform one complete Monte Carlo sweep over all molecules
    """

    def __init__(self, box, T: float, b: float, *, log_energy: bool = True) -> None:
        """
        Initialize the Metropolis Monte Carlo simulation.
        
        Parameters
        ----------
        box : Box
            The simulation box containing molecules and potential
        T : float
            Temperature of the simulation in reduced units
        b : float
            Maximum displacement parameter for random moves
        log_energy : bool
            Whether to track potential energy during simulation (default is True)
        """
        super().__init__(box, log_energy=log_energy)
        self.T = T
        self.b = b

    def step(self) -> float:
        """
        Perform one Monte Carlo sweep using the Metropolis algorithm.
        
        Attempts to move each molecule once on average by selecting random
        molecules and applying the Metropolis acceptance criterion. The
        energy difference is calculated and moves are accepted based on
        the Boltzmann probability exp(-ΔE/T).
        
        Returns
        -------
        float
            Acceptance ratio for this Monte Carlo step (accepted moves / total attempts)
        """
        accepted = 0
        N = len(self.box._molecules)
        for _ in range(N):
            idx = np.random.randint(N)
            mol = self.box._molecules[idx]
            
            # Calculate before the move
            energies_before = []
            for other in self.box.get_molecules:
                if other is not mol:
                    energy = self.box.potential.potential_energy(
                        mol.position, other.position, self.box.box_size
                    )
                    energies_before.append(energy)
            energy_before_move = sum(energies_before)

            # Perform a random trial move
            
            # Calculate before the move
            energies_before = []
            for other in self.box.get_molecules:
                if other is not mol:
                    energy = self.box.potential.potential_energy(
                        mol.position, other.position, self.box.box_size
                    )
                    energies_before.append(energy)
            energy_before_move = sum(energies_before)

            # Perform a random trial move
            mol.move_random(self.b, self.box.box_size)

            # Calculate after the move
            energies_after = []
            for other in self.box.get_molecules:
                if other is not mol:
                    energy = self.box.potential.potential_energy(
                        mol.alt_position, other.position, self.box.box_size
                    )
                    energies_after.append(energy)
            energy_after_move = sum(energies_after)

            # Compute the change in potential energy
            delta_E = energy_after_move - energy_before_move


            # Calculate after the move
            energies_after = []
            for other in self.box.get_molecules:
                if other is not mol:
                    energy = self.box.potential.potential_energy(
                        mol.alt_position, other.position, self.box.box_size
                    )
                    energies_after.append(energy)
            energy_after_move = sum(energies_after)

            # Compute the change in potential energy
            delta_E = energy_after_move - energy_before_move

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
    
    The Orchestrator manages the entire simulation workflow including
    configuration loading, box setup, simulation initialization, and
    execution control. It reads parameters from JSON configuration files
    and handles output formatting and file management.
    
    Attributes
    ----------
    config_file : str
        Path to the JSON configuration file
    config : dict
        Loaded configuration parameters
    box : Box or None
        The simulation box instance
    simulation : Simulation or None
        The simulation instance
    current_step : int
        Current simulation step number
    
    Methods
    -------
    setup_box(box_class)
        Initialize the simulation box based on configuration
    setup_simulation(simulation_class, **kwargs)
        Initialize the simulation instance
    run_simulation()
        Execute the complete simulation workflow
    print_config_summary()
        Display a summary of loaded configuration parameters
    """
    
    def __init__(self, config_file: str) -> None:
        """
        Initialize the orchestrator with configuration file.
        
        Parameters
        ----------
        config_file : str
            Path to the JSON configuration file containing simulation parameters
            
        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be found
        ValueError
            If the configuration file contains invalid JSON
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.box = None
        self.simulation = None
        self.current_step = 0
        
    def _load_config(self) -> dict:
        """
        Load configuration parameters from JSON file.
        
        Returns
        -------
        dict
            Dictionary containing all configuration parameters
            
        Raises
        ------
        FileNotFoundError
            If the configuration file cannot be found
        ValueError
            If the JSON file contains syntax errors
        """
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def setup_box(self, box_class: type) -> None:
        """
        Initialize the simulation box based on configuration parameters.
        
        Creates a box instance with dimensions specified in the configuration
        and sets up compartments if defined in the configuration file.
        
        Parameters
        ----------
        box_class : type
            The Box class to instantiate for the simulation
        """
        setup_config = self.config["setup"]
        box_size = np.array([setup_config["Lx"], setup_config["Ly"], setup_config["Lz"]])
        
        # Create box instance
        self.box = box_class(box_size)
        
        # Setup compartments if specified
        if "compartments" in setup_config:
            self._setup_compartments(setup_config["compartments"])
    
    def _setup_compartments(self, compartments: list) -> None:
        """
        Initialize compartments in the simulation box.
        
        Sets up different regions within the box with specified densities
        and volume fractions based on the configuration parameters.
        
        Parameters
        ----------
        compartments : list of dict
            List of compartment configurations, each containing density
            and volume_fraction parameters
        """
        # This method would need to be implemented based on your Box class
        # compartment setup functionality
        for i, compartment in enumerate(compartments):
            density = compartment["density"]
            volume_fraction = compartment["volume_fraction"]
            # Add molecules to compartment based on density and volume_fraction
            # Implementation depends on your Box class methods
            print(f"Setting up compartment {i+1}: density={density}, volume_fraction={volume_fraction}")
    
    def setup_simulation(self, simulation_class: type, **simulation_kwargs: Any) -> None:
        """
        Initialize the simulation instance with specified parameters.
        
        Creates a simulation object using the provided class and merges
        configuration parameters with any additional keyword arguments.
        
        Parameters
        ----------
        simulation_class : type
            The Simulation class to instantiate
        **simulation_kwargs : dict
            Additional keyword arguments for simulation initialization
            
        Raises
        ------
        RuntimeError
            If the box has not been set up before calling this method
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
    
    def run_simulation(self) -> None:
        """
        Execute the complete simulation workflow.
        
        Runs the Monte Carlo simulation according to the configuration
        parameters, handling console output, trajectory writing, and
        configuration file output at specified intervals.
        """
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
    
    def print_config_summary(self) -> None:
        """
        Display a formatted summary of the loaded configuration parameters.
        
        Prints key simulation parameters including box dimensions, step counts,
        output frequencies, and compartment information in a readable format.
        """
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


class SimulationFactory:
    """
    Factory class for registering and creating Simulation instances.
    
    The SimulationFactory implements the factory design pattern to provide
    a centralized registry for different simulation types. It allows dynamic
    registration of simulation classes and creation of simulation instances
    by name, promoting modularity and extensibility in the simulation framework.
    
    Attributes
    ----------
    _types : dict
        Dictionary mapping simulation type names to their corresponding classes
    
    Methods
    -------
    register(name, simulation_class)
        Register a new simulation class with a given name
    __call__(name, **kwargs)
        Create and return a simulation instance of the specified type
    """
    
    def __init__(self) -> None:
        """
        Initialize the simulation factory with an empty registry.
        
        Creates an empty dictionary to store the mapping between
        simulation type names and their corresponding classes.
        """
        self._types = {}

    def register(self, name: str, simulation_class: type) -> None:
        """
        Register a simulation class with the factory under a given name.
        
        Adds a new simulation class to the factory registry, allowing it
        to be instantiated later using the provided name. The class must
        inherit from the Simulation base class.
        
        Parameters
        ----------
        name : str
            The name identifier for the simulation type
        simulation_class : type
            The simulation class to register, must inherit from Simulation
            
        Raises
        ------
        TypeError
            If the provided class does not inherit from Simulation
        """
        if not issubclass(simulation_class, Simulation):
            raise TypeError(f"{simulation_class} must inherit from Simulation")
        self._types[name] = simulation_class

    def __call__(self, name: str, **kwargs: Any) -> Simulation:
        """
        Create and return a simulation instance of the specified type.
        
        Instantiates a simulation object using the class registered under
        the given name, passing all keyword arguments to the constructor.
        This method makes the factory instance callable.
        
        Parameters
        ----------
        name : str
            The name of the registered simulation type to create
        **kwargs
            Keyword arguments to pass to the simulation constructor
            
        Returns
        -------
        Simulation
            An instance of the requested simulation type
            
        Raises
        ------
        KeyError
            If the requested simulation type name is not registered
        """
        if name not in self._types:
            raise KeyError(f"Unknown simulation type: {name}")

        return self._types[name](**kwargs)

    

    


class SimulationFactory:
    """
    Factory class for registering and creating Simulation instances.
    
    The SimulationFactory implements the factory design pattern to provide
    a centralized registry for different simulation types. It allows dynamic
    registration of simulation classes and creation of simulation instances
    by name, promoting modularity and extensibility in the simulation framework.
    
    Attributes
    ----------
    _types : dict
        Dictionary mapping simulation type names to their corresponding classes
    
    Methods
    -------
    register(name, simulation_class)
        Register a new simulation class with a given name
    __call__(name, **kwargs)
        Create and return a simulation instance of the specified type
    """
    
    def __init__(self) -> None:
        """
        Initialize the simulation factory with an empty registry.
        
        Creates an empty dictionary to store the mapping between
        simulation type names and their corresponding classes.
        """
        self._types = {}

    def register(self, name: str, simulation_class: type) -> None:
        """
        Register a simulation class with the factory under a given name.
        
        Adds a new simulation class to the factory registry, allowing it
        to be instantiated later using the provided name. The class must
        inherit from the Simulation base class.
        
        Parameters
        ----------
        name : str
            The name identifier for the simulation type
        simulation_class : type
            The simulation class to register, must inherit from Simulation
            
        Raises
        ------
        TypeError
            If the provided class does not inherit from Simulation
        """
        if not issubclass(simulation_class, Simulation):
            raise TypeError(f"{simulation_class} must inherit from Simulation")
        self._types[name] = simulation_class

    def __call__(self, name: str, **kwargs: Any) -> Simulation:
        """
        Create and return a simulation instance of the specified type.
        
        Instantiates a simulation object using the class registered under
        the given name, passing all keyword arguments to the constructor.
        This method makes the factory instance callable.
        
        Parameters
        ----------
        name : str
            The name of the registered simulation type to create
        **kwargs
            Keyword arguments to pass to the simulation constructor
            
        Returns
        -------
        Simulation
            An instance of the requested simulation type
            
        Raises
        ------
        KeyError
            If the requested simulation type name is not registered
        """
        if name not in self._types:
            raise KeyError(f"Unknown simulation type: {name}")

        return self._types[name](**kwargs)

    

    


class SimulationFactory:
    """
    Factory to register and create Potential instances.
    """
    def __init__(self):
        self._types = {}

    def register(self, name: str, simulation_class: type):
        if not issubclass(simulation_class, Simulation):
            raise TypeError(f"{simulation_class} must inherit from Potential")
        self._types[name] = simulation_class

    def __call__(self, name: str, **kwargs) -> Simulation:
        if name not in self._types:
            raise KeyError(f"Unknown potential type: {name}")
        return self._types[name](**kwargs)