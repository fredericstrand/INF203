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
from src.timeseries.box_average import box_average


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
        with open(config_file, "r") as jf:
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
            potential=potential,
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
        result_file = cfg.get("results_output", {}).get("file")

        if not result_file:
            raise KeyError("No results output file specified in configuration")

        conf_cfg = cfg.get("configuration_output", {})
        init_file = conf_cfg.get("initial")

        final_file = conf_cfg.get("final")
        ctrl = cfg["control_parameters"]
        max_disp = ctrl["maximum_displacement"]
        distortions = ctrl["test_area_distortion"]
        d1, d2 = distortions[0], distortions[1]
        sx1, sy1, sz1 = d1["sx"], d1["sy"], d1["sz"]
        sx2, sy2, sz2 = d2["sx"], d2["sy"], d2["sz"]

        if not (np.isclose(sx1 * sy1 * sz1, 1.0) and np.isclose(sx2 * sy2 * sz2, 1.0)):
            raise ValueError(
                "Distortions must be volume-conserving (sx * sy * sz = 1.0)"
            )

        zeta = sx1 * sz1
        sqrt_zeta = zeta**0.5

        # Print initial stats and save initial configuration
        print(f"Initial # molecules: {len(box._molecules)}")
        print(f"Initial E_pot:        {box.total_epot:.5f}")
        if init_file:
            os.makedirs(os.path.dirname(init_file), exist_ok=True)
            box.write_XYZ(init_file)

        # Setup the Metropolis Monte Carlo simulation
        orchestrator.setup_simulation(MetropolisMC, T=T, log_energy=True)
        mc = orchestrator.simulation
        setattr(mc, "b", max_disp)

        # Prepare logging
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Open log file and write parameter header
        exp_s1, exp_s2 = [], []
        with open(result_file, "w") as f_log:
            f_log.write("# Simulation Parameters\n")
            f_log.write(f"# Box dimensions: {Lx} x {Ly} x {Lz}\n")
            f_log.write(f"# Temperature: {T}\n")
            f_log.write(f"# Total steps: {n_pr}\n")
            f_log.write(f"# Log interval: {log_interval}\n")
            f_log.write(f"# Max displacement: {max_disp}\n")
            f_log.write(
                f"# Distortions: d1=({sx1},{sy1},{sz1}), d2=({sx2},{sy2},{sz2})\n"
            )
            f_log.write(f"# Reset sampling at: {sorted(reset_points)}\n")
            f_log.write(f"# Console freq: {console_freq}\n")
            f_log.write(f"# Trajectory freq/file: {log_interval}/{traj_file}\n")
            f_log.write(
                "\n# step   E_pot     acc     w1        avg1      gamma1    w2        avg2      gamma2\n"
            )

            # Run the main Monte Carlo loop
            for step in range(1, n_pr + 1):
                acc = mc.step()
                Epot = box.total_epot

                # Reset accumulators if requested
                if step in reset_points:
                    exp_s1.clear()
                    exp_s2.clear()

                # Compute distortions each MC step
                dU1, dA1 = compute_distortion(box, sx1, sy1, sz1)
                dU2, dA2 = compute_distortion(box, sx2, sy2, sz2)

                # Boltzmann weights
                w1 = np.exp(-dU1 / T)
                w2 = np.exp(-dU2 / T)
                exp_s1.append(w1)
                exp_s2.append(w2)

                # Console output
                if console_freq and (step % console_freq == 0):
                    print(f"Step {step}: E_pot={Epot:.3f}, acc={acc:.3f}")

                # Write trajectory to file
                if traj_file and (step % log_interval == 0):
                    os.makedirs(os.path.dirname(traj_file), exist_ok=True)
                    box.write_XYZ(traj_file)

                # Logging to file at intervals
                if step % log_interval == 0:
                    avg1 = np.mean(exp_s1)
                    avg2 = np.mean(exp_s2)
                    gamma1 = -T * np.log(avg1) / dA1
                    gamma2 = -T * np.log(avg2) / dA2

                    f_log.write(
                        f"{step:6d}  {Epot:8.3f}  {acc:5.3f}  "
                        f"{w1:7.3e}  {avg1:7.3e}  {gamma1:8.3f}  "
                        f"{w2:7.3e}  {avg2:7.3e}  {gamma2:8.3f}\n"
                    )

            # Write final surface tension estimates
            f_log.write("\n")
            f_log.write(f"gamma(area increase) = {gamma1:.6f}\n")
            f_log.write(f"gamma(area decrease) = {gamma2:.6f}\n")

        # adding uncertanty analysis to log file
        mean, std = box_average(result_file)
        print("\n=== Uncertainty analysis ===")
        print(f"mean: {mean}")
        print(f"std: {std}")
        with open(result_file, "a") as f_log:
            f_log.write(f"mean: {mean}\n")
            f_log.write(f"std: {std}\n")

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
