from src.ljts.box import Box

box = Box(5, 40, 5)
box.populate_box(0.73, 0.02)

print(f"Number of molecules: {len(box.molecules)}")
