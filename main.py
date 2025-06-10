import numpy as np
from src.ljts.potential import LJTS
from src.ljts.box import Box
from src.ljts.simulation import MetropolisMC
from src.ljts.distortion import compute_distortion
from src.config import parseArgs
import os


def main():
    args = parseArgs()
    os.makedirs("data", exist_ok=True)
    # os.makedirs(os.path.dirname(args.log), exist_ok=True)
    
    # 1) Create the inter‐particle potential
    potential = LJTS(cutoff=2.5)
    box = Box(
        len_x=5,
        len_y=40,
        len_z=5,
        den_liq=0.73,
        den_vap=0.02,
        potential=potential
    )
    
    # 3) Optionally print initial stats
    print(f"Initial # molecules: {len(box._molecules)}")
    print(f"Initial E_pot:        {box.total_epot:.5f}")
    box.write_XYZ("data/config_init.xyz", mode="w")

    # change these variables to increase accuracy
    T = 0.8
    zeta = 1.01
    n_eq = 1000
    n_pr = 2000
    log_interval = 500

    mc = MetropolisMC(box, T=T, b=1/8, log_energy=False)
    mc.run(n_steps=n_eq, log_interval=log_interval)

    exp_s1 = []
    exp_s2 = []

    sqrt_zeta = zeta ** 0.5

    for step in range(1, n_pr + 1):
        acceptance = mc.step()
        Epot = box.total_epot

        if step % log_interval == 0:
            # distortion that increases area
            dU1, dA1 = compute_distortion(
                box.get_molecules,
                box.box_size,
                box.potential, 
                sqrt_zeta,
                1.0 / zeta,
                sqrt_zeta
            )
            # distortion that decreases area
            dU2, dA2 = compute_distortion(
                box.get_molecules,
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

            print(f"[Step {step}] Epot={Epot:.3f} acc={acceptance:.3f} e1={e1:.3e} avg1={avg1:.3e} gamma1={gamma1:.3f} | e2={e2:.3e} avg2={avg2:.3e} gamma2={gamma2:.3f}")

    print(
        f"gamma(area increase) = {gamma1:.6f}, gamma(area decrease) = {gamma2:.6f}"
    )

if __name__ == "__main__":

    main()
