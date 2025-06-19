from typing import Optional
import numpy as np

from src.ljts.potential import Potential

def compute_distortion(molecules: list, box_size: np.ndarray, potential: Optional[Potential], sx: float, sy: float, sz: float) -> tuple[float, float]:
    """
    Compute the change in potential energy and interface area for a volume-conserving distortion.
    
    Calculates the change in potential energy (delta U) and interface area (delta A)
    for a small, volume-conserving distortion of the simulation box and coordinates
    by scaling the box dimensions and molecule positions.
    
    Parameters
    ----------
    molecules : list
        List of molecule objects containing position information
    box_size : numpy.ndarray
        Original box dimensions as [len_x, len_y, len_z]
    potential : object
        Potential energy calculation object
    sx : float
        Scaling factor for x-direction
    sy : float
        Scaling factor for y-direction
    sz : float
        Scaling factor for z-direction
        
    Returns
    -------
    tuple of float
        A tuple containing (delta_U, delta_A) where delta_U is the change
        in potential energy and delta_A is the change in interface area
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
