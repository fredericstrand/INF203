import numpy as np

def compute_delta_U_and_A(molecules, box_size, potential, sx, sy, sz):
    """
    Compute the change in potential energy (ΔU) and interface area (ΔA)
    for a small, volume-conserving distortion of the simulation box and coordinates.
    """
    # Undistorted energy
    E0 = 0.0
    N = len(molecules)
    for i in range(N):
        for j in range(i + 1, N):
            p1 = molecules[i].position
            p2 = molecules[j].position
            E0 += potential.potential_energy(p1, p2, box_size)

    # Distorted box size and scale matrix
    new_box = box_size * np.array([sx, sy, sz])
    scale_matrix = np.diag([sx, sy, sz])

    # Distorted energy
    E1 = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            p1 = scale_matrix @ molecules[i].position
            p2 = scale_matrix @ molecules[j].position
            E1 += potential.potential_energy(p1, p2, new_box)

    delta_U = E1 - E0

    # Interface area change
    A0 = 2 * box_size[0] * box_size[2]
    A1 = 2 * new_box[0]   * new_box[2]
    delta_A = A1 - A0

    return delta_U, delta_A