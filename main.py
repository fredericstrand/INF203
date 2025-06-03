from src.ljts.box import Box

box = Box(5, 40, 5, den_liq=0.73, den_vap=0.02)
box.total_potential_energy()
print(f"total potential energy: {box._total_Epot}")
print(f"Number of molecules: {len(box._molecules)}")
