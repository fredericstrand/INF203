#!/usr/bin/env python3
"""
Main script for running Monte Carlo simulations.
Can run with JSON configuration or use legacy mode.
Usage: 
  python main.py <config_file.json>  # JSON configuration mode
  python main.py                     # Legacy mode
"""

import sys
import os
from src.ljts.potential import LJTS
from src.ljts.box import Box
from src.ljts.orchestrator import Orchestrator, MetropolisMC
from src.config import parseArgs


def run_with_orchestrator(config_file: str):
    """Run simulation using Orchestrator with JSON configuration."""
    try:
        # Create orchestrator with configuration file
        orchestrator = Orchestrator(config_file)
        orchestrator.print_config_summary()
        
        # Create the inter-particle potential
        potential = LJTS(cutoff=2.5)
        
        # Get box dimensions from config
        setup = orchestrator.config["setup"]
        box = Box(
            len_x=setup["Lx"], 
            len_y=setup["Ly"], 
            len_z=setup["Lz"], 
            den_liq=0.73,  # Default values, could be made configurable
            den_vap=0.02,
            potential=potential
        )
        
        # Set the box in orchestrator
        orchestrator.box = box
        
        # Print initial stats
        print(f"Initial # molecules: {len(box._molecules)}")
        print(f"Initial E_pot:        {box.total_epot:.5f}")
        
        # Setup simulation with parameters from config
        T = 0.8  # Could be made configurable via JSON
        max_displacement = orchestrator.config["control_parameters"]["maximum_displacement"]
        
        orchestrator.setup_simulation(
            MetropolisMC,
            T=T,
            log_energy=True
        )
        
        # Run the full simulation as configured
        orchestrator.run_simulation()
        
        # Final results
        print("\n=== Final Results ===")
        print(f"Final E_pot:        {box.total_epot:.5f}")
        print(f"Total # molecules:  {len(box._molecules)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def run_legacy_mode():
    """Run simulation using the original legacy approach."""
    args = parseArgs()
    os.makedirs("data", exist_ok=True)
    
    # 1) Create the inter-particle potential
    potential = LJTS(cutoff=2.5)
    
    # 2) Build the Box and populate it
    box = Box(
        len_x=5, len_y=40, len_z=5, den_liq=0.73, den_vap=0.02, potential=potential
    )
    
    # 3) Optionally print initial stats
    print(f"Initial # molecules: {len(box._molecules)}")
    print(f"Initial E_pot:        {box.total_epot:.5f}")
    box.write_XYZ("data/config_init.xyz", mode="w")
    
    # 4) Set up Monte Carlo simulation
    mc = MetropolisMC(
        box,  # your Box instance
        T=0.8,  # temperature
        b=1 / 8,  # max displacement
        log_energy=True,
    )
    
    # 5) Equilibration phase
    print("\n=== Equilibration ===")
    mc.run(n_steps=1000, log_interval=200)
    
    # 6) Production phase
    print("\n=== Production ===")
    mc.run(n_steps=2000, log_interval=200)
    
    # 7) Final results
    print("\n=== Final Results ===")
    print(f"Final E_pot:        {box.total_epot:.5f}")
    print(f"Total # molecules:  {len(box._molecules)}")
    box.write_XYZ("data/config_final.xyz", mode="w")


def main():
    """Main function - determines whether to use JSON config or legacy mode."""
    # Check if a JSON config file was provided as command line argument
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        config_file = sys.argv[1]
        print(f"Running with JSON configuration: {config_file}")
        run_with_orchestrator(config_file)
    # else:
    #     print("Running in legacy mode (no JSON config provided)")
    #     run_legacy_mode()


if __name__ == "__main__":
    main()
