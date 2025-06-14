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
    
    # Create the inter‐particle potential & box
    potential = LJTS(cutoff=2.5)
    box = Box(
        len_x=5,
        len_y=40,
        len_z=5,
        den_liq=0.73,
        den_vap=0.02,
        potential=potential
    )
    
    # Parameters you might even pull from args
    T            = 0.8
    zeta         = 1.01
    n_eq         = 1000
    n_pr         = 2000
    log_interval = 500
    
    # Open file and write header
    with open(f"results/{T}_{zeta}_{n_pr}", "w") as f:
        f.write("# run parameters\n")
        f.write(f"T           = {T}\n")
        f.write(f"zeta        = {zeta}\n")
        f.write(f"n_eq        = {n_eq}\n")
        f.write(f"n_pr        = {n_pr}\n")
        f.write(f"log_interval= {log_interval}\n")
        f.write("\n# step  Epot    acc    e1      avg1    gamma1   e2      avg2    gamma2\n")
        
        # Initial epot
        f.write(f"initial  {box.total_epot:.5f}  -      -       -       -       -       -      -\n")
        
        # Run MC and log into file
        mc = MetropolisMC(box, T=T, b=1/8, log_energy=False)
        mc.run(n_steps=n_eq, log_interval=log_interval)
    
        exp_s1 = []
        exp_s2 = []
        sqrt_zeta = zeta**0.5
    
        for step in range(1, n_pr + 1):
            acc   = mc.step()
            Epot  = box.total_epot
    
            if step % log_interval == 0:
                # compute distortions
                dU1, dA1 = compute_distortion(
                    box.get_molecules,
                    box.box_size,
                    box.potential, 
                    sqrt_zeta,
                    1.0 / zeta,
                    sqrt_zeta
                )
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
    
                avg1   = np.mean(exp_s1)
                avg2   = np.mean(exp_s2)
                gamma1 = -T * np.log(avg1) / dA1
                gamma2 = -T * np.log(avg2) / dA2
    
                f.write(
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
                
        f.write("\n")
        f.write(f"gamma(area increase) = {gamma1:.6f}\n")
        f.write(f"gamma(area decrease) = {gamma2:.6f}\n")

if __name__ == "__main__":
    main()
