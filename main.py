from src.ljts.box import Box

box = Box(5, 40, 5)
box.populate_box(0.73, 0.02)
box.total_potential_energy()
print(f"total potential energy: {box.total_Epot}")
print(f"Number of molecules: {len(box.molecules)}")
