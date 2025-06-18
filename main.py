import sys
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
            den_liq=setup["compartments"][1]["density"],
            den_vap=setup["compartments"][0]["density"],
            potential=potential
        )
        
        # Set the box in orchestrator
        orchestrator.box = box
        
        # Print initial stats
        print(f"Initial # molecules: {len(box._molecules)}")
        print(f"Initial E_pot:        {box.total_epot:.5f}")
        
        T = 0.8
        
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


def main():
    args = parseArgs()
    config_file = args.file
    log_file = args.log
    
    print(f"Running with JSON configuration: {config_file}")
    print(f"Logging to: {log_file}")
    
    run_with_orchestrator(config_file)


if __name__ == "__main__":
    main()
