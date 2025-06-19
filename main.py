"""Monte Carlo Surface Tension Simulator

This script runs a Monte Carlo simulation of a Lennard-Jones Truncated-Shifted (LJTS)
fluid in a rectangular simulation box to compute the surface tension using the test-area 
method. It supports input via a JSON configuration file describing system, simulation, 
and output parameters.

This script uses modules for defining molecular potentials, simulation boxes, orchestration,
Monte Carlo moves, and distortion-based surface tension estimation.

The simulator performs the following steps:
    * Reads simulation parameters from a JSON config file
    * Initializes the molecular system and Monte Carlo engine
    * Performs a loop of Monte Carlo steps with energy and surface tension tracking
    * Outputs periodic logs and final statistics to file
    * Optionally saves initial and final molecular configurations in XYZ format

Required packages:
    * numpy
    * The local `src.ljts` and `src.config` modules for domain-specific simulation logic

This file can also be imported as a module and contains the following functions:

    * run_with_orchestrator - runs the main Monte Carlo simulation with full configuration
    * main - entry point that parses arguments and launches the simulation
"""

import sys
import os
import json
import numpy as np

from src.ljts.potential import LJTS
from src.ljts.box import Box
from src.ljts.orchestrator import Orchestrator, MetropolisMC
from src.ljts.distortion import compute_distortion
from src.config import parseArgs


def run_with_orchestrator(config_file: str):
    """Run the surface tension Monte Carlo simulation from a JSON configuration file.

    This function reads parameters from a JSON configuration file, initializes the 
    molecular box and simulation engine, and performs a Metropolis Monte Carlo run
    while logging thermodynamic quantities including surface tension estimates.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file describing setup, control, and output.
    """
    try:
        # Load configuration from file
        with open(config_file, 'r') as jf:
            cfg = json.load(jf)

        # Initialize orchestrator and display configuration
        orchestrator = Orchestrator(config_file)
        orchestrator.print_config_summary()

        # Unpack simulation box parameters
        setup = cfg["setup"]
        Lx, Ly, Lz = setup["Lx"], setup["Ly"], setup["Lz"]
        T = setup.get("temperature", 0.8)

        # Initialize potential and box
        potential = LJTS(cutoff=2.5)
        box = Box(
            len_x=Lx,
            len_y=Ly,
            len_z=Lz,
            den_liq=setup["compartments"][1]["density"],
            den_vap=setup["compartments"][0]["density"],
            potential=potential
        )
        orchestrator.box = box

        # Extract simulation control parameters
        sim_cfg = cfg["steps"]
        n_pr = sim_cfg["total"]
        reset_points = set(sim_cfg.get("reset_sampling_at", []))

        # Output configuration
        console_freq = cfg.get("console_output", {}).get("frequency")
        traj_cfg = cfg.get("trajectory_output", {})
        log_interval = traj_cfg.get("frequency")
        traj_file = traj_cfg.get("file")

        conf_cfg = cfg.get("configuration_output", {})
        init_file = conf_cfg.get("initial")
        final_file = conf_cfg.get("final")

        ctrl = cfg["control_parameters"]
        max_disp = ctrl["maximum_displacement"]
        distortions = ctrl["test_area_distortion"]
        d1, d2 = distortions[0], distortions[1]
        zeta = d2["sx"]
        sqrt_zeta = zeta ** 0.5

        # Print initial stats and save initial configuration
        print(f"Initial # molecules: {len(box._molecules)}")
        print(f"Initial E_pot:        {box.total_epot:.5f}")
        if init_file:
            os.makedirs(os.path.dirname(init_file), exist_ok=True)
            box.write_XYZ(init_file)

        # Setup the Metropolis Monte Carlo simulation
        orchestrator.setup_simulation(
            MetropolisMC,
            T=T,
            log_energy=False
        )
        mc = orchestrator.simulation
        setattr(mc, "b", max_disp)

        # Prepare logging
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        param_str = f"steps{n_pr}_{zeta}"
        log_fname = os.path.join(results_dir, f"result_{param_str}.log")

        # Open log file and write parameter header
        exp_s1, exp_s2 = [], []
        with open(log_fname, "w") as f_log:
            f_log.write("# Simulation Parameters\n")
            f_log.write(f"# Box dimensions: {Lx} x {Ly} x {Lz}\n")
            f_log.write(f"# Temperature: {T}\n")
            f_log.write(f"# Total steps: {n_pr}\n")
            f_log.write(f"# Log interval: {log_interval}\n")
            f_log.write(f"# Max displacement: {max_disp}\n")
            f_log.write(f"# Distortions (d1, d2): {d1}, {d2}\n")
            f_log.write(f"# Reset sampling at: {sorted(reset_points)}\n")
            f_log.write(f"# Console freq: {console_freq}\n")
            f_log.write(f"# Trajectory freq/file: {log_interval}/{traj_file}\n")
            f_log.write("\n# step   E_pot     acc     e1        avg1      gamma1    "
                        "e2        avg2      gamma2\n")

            # Run the main Monte Carlo loop
            for step in range(1, n_pr + 1):
                acc = mc.step()
                Epot = box.total_epot

                if step in reset_points:
                    exp_s1.clear()
                    exp_s2.clear()

                # Console output
                if console_freq and (step % console_freq == 0):
                    print(f"Step {step}: E_pot={Epot:.3f}, acc={acc:.3f}")

                # Write trajectory to file
                if traj_file and (step % log_interval == 0):
                    os.makedirs(os.path.dirname(traj_file), exist_ok=True)
                    box.write_XYZ(traj_file)

                # Compute surface tension
                if step % log_interval == 0:
                    dU1, dA1 = compute_distortion(
                        box._molecules,
                        box.box_size,
                        box.potential,
                        sqrt_zeta,
                        1.0 / zeta,
                        sqrt_zeta
                    )
                    dU2, dA2 = compute_distortion(
                        box._molecules,
                        box.box_size,
                        box.potential,
                        1.0 / sqrt_zeta,
                        zeta,
                        1.0 / sqrt_zeta
                    )

                    e1 = np.exp(-dU1 / T)
                    e2 = np.exp(-dU2 / T)
                    exp_s1.append(e1)
                    exp_s2.append(e2)

                    avg1 = np.mean(exp_s1)
                    avg2 = np.mean(exp_s2)
                    gamma1 = -T * np.log(avg1) / dA1
                    gamma2 = -T * np.log(avg2) / dA2

                    f_log.write(
                        f"{step:6d}  "
                        f"{Epot:8.3f}  "
                        f"{acc:5.3f}  "
                        f"{e1:7.3e}  "
                        f"{avg1:7.3e}  "
                        f"{gamma1:8.3f}  "
                        f"{e2:7.3e}  "
                        f"{avg2:7.3e}  "
                        f"{gamma2:8.3f}\n"
                    )

            # Write final surface tension estimates
            f_log.write("\n")
            f_log.write(f"gamma(area increase) = {gamma1:.6f}\n")
            f_log.write(f"gamma(area decrease) = {gamma2:.6f}\n")

        # Save final configuration
        if final_file:
            os.makedirs(os.path.dirname(final_file), exist_ok=True)
            box.write_XYZ(final_file)

        # Print final summary
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
    """Main function: parses command-line arguments and runs simulation."""
    args = parseArgs()
    config_file = args.file
    log_file = args.log

    print(f"Running with JSON configuration: {config_file}")
    print(f"Logging to: {log_file}")

    run_with_orchestrator(config_file)


if __name__ == "__main__":
    main()
