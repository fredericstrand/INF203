import sys
import os
import json
import numpy as np

from src.ljts.potential import LJTS
from src.ljts.box import Box
from src.ljts.orchestrator import Orchestrator, MetropolisMC
from src.ljts.distortion import compute_distortion


def run_with_orchestrator(config_file: str):
    """Run simulation using Orchestrator with JSON configuration."""
    try:
        # Load JSON config directly for parameter logging
        with open(config_file, 'r') as jf:
            cfg = json.load(jf)

        # Instantiate orchestrator & print its summary
        orchestrator = Orchestrator(config_file)
        orchestrator.print_config_summary()

        # Extract setup parameters
        setup = cfg["setup"]
        Lx, Ly, Lz = setup["Lx"], setup["Ly"], setup["Lz"]
        T = setup.get("temperature", 0.8)

        # Build potential & box
        potential = LJTS(cutoff=2.5)
        box = Box(
            len_x   = Lx,
            len_y   = Ly,
            len_z   = Lz,
            den_liq = setup["compartments"][1]["density"],
            den_vap = setup["compartments"][0]["density"],
            potential=potential
        )
        orchestrator.box = box

        # Extract step parameters
        sim_cfg      = cfg["steps"]
        n_pr         = sim_cfg["total"]
        reset_points = set(sim_cfg.get("reset_sampling_at", []))

        # Extract console + trajectory output parameters
        console_freq = cfg.get("console_output", {}).get("frequency")
        traj_cfg     = cfg.get("trajectory_output", {})
        log_interval = traj_cfg.get("frequency")
        traj_file    = traj_cfg.get("file")

        # Extract configuration output parameters
        conf_cfg   = cfg.get("configuration_output", {})
        init_file  = conf_cfg.get("initial")
        final_file = conf_cfg.get("final")

        # Extract control parameters & distortions
        ctrl        = cfg["control_parameters"]
        max_disp    = ctrl["maximum_displacement"]
        distortions = ctrl["test_area_distortion"]
        d1, d2      = distortions[0], distortions[1]
        sx1, sy1, sz1 = d1["sx"], d1["sy"], d1["sz"]
        sx2, sy2, sz2 = d2["sx"], d2["sy"], d2["sz"]

        # Initial stats & initial config dump
        print(f"Initial # molecules: {len(box._molecules)}")
        print(f"Initial E_pot:        {box.total_epot:.5f}")
        if init_file:
            os.makedirs(os.path.dirname(init_file), exist_ok=True)
            box.write_XYZ(init_file)

        # Set up Metropolis MC
        orchestrator.setup_simulation(
            MetropolisMC,
            T=T,
            log_energy=True
        )
        mc = orchestrator.simulation
        setattr(mc, "b", max_disp)

        # Prepare results directory & log filename with parameters
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        param_str = f"steps{n_pr}_d1({sx1},{sy1},{sz1})"
        log_fname = os.path.join(results_dir, f"result_{param_str}.log")

        # Open log and write parameters header
        exp_s1, exp_s2 = [], []
        with open(log_fname, "w") as f_log:
            f_log.write("# Simulation Parameters\n")
            f_log.write(f"# Box dimensions: {Lx} x {Ly} x {Lz}\n")
            f_log.write(f"# Temperature: {T}\n")
            f_log.write(f"# Total steps: {n_pr}\n")
            f_log.write(f"# Log interval: {log_interval}\n")
            f_log.write(f"# Max displacement: {max_disp}\n")
            f_log.write(f"# Distortions: d1=({sx1},{sy1},{sz1}), d2=({sx2},{sy2},{sz2})\n")
            f_log.write(f"# Reset sampling at: {sorted(reset_points)}\n")
            f_log.write(f"# Console freq: {console_freq}\n")
            f_log.write(f"# Trajectory freq/file: {log_interval}/{traj_file}\n")
            f_log.write("\n# step   E_pot     acc     w1        avg1      gamma1    w2        avg2      gamma2\n")

            # Monte Carlo loop with distortion logging
            for step in range(1, n_pr + 1):
                acc  = mc.step()
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

                # Trajectory dump
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

            # Final surface-tension summary
            f_log.write("\n")
            f_log.write(f"gamma(area increase) = {gamma1:.6f}\n")
            f_log.write(f"gamma(area decrease) = {gamma2:.6f}\n")

        # Final configuration dump
        if final_file:
            os.makedirs(os.path.dirname(final_file), exist_ok=True)
            box.write_XYZ(final_file)

        # Final console output
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
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        print(f"Running with JSON configuration: {sys.argv[1]}")
        run_with_orchestrator(sys.argv[1])
    else:
        print("Usage: python main.py <config.json>", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()