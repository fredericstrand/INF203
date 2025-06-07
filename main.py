from src.ljts.box import Box


box = Box(5, 40, 5, den_liq=0.73, den_vap=0.02)
for step in range(1, 1001):
    acceptance = box.simulation(T=0.8, b=1 / 8)

    if step % 50 == 0 or step == 1:
        print(f"{step} E_pot: {box.get_total_epot} Acceptance: {acceptance:.2f}")


box.total_potential_energy()
print(f"total potential energy: {box._total_Epot}")
print(f"Number of molecules: {len(box._molecules)}")
