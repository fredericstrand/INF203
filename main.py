from src.ljts.potential import LJTS
from src.ljts.box import Box
from src.ljts.simulation import MetropolisMC
from src.config import parseArgs
import os


def main():

    args = parseArgs()
    os.makedirs("data", exist_ok=True)
    # os.makedirs(os.path.dirname(args.log), exist_ok=True)
    # 1) Create the inter‐particle potential
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


if __name__ == "__main__":

    main()
