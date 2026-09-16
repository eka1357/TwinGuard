from typing import Sequence

import numpy as np


def get_plate_position(sim) -> np.ndarray:
    """Return the plate world position as an XYZ NumPy array."""
    plate_state = sim.get_joint_positions()["plate_joint"]
    return np.asarray(plate_state[:3], dtype=float)


def gripper_to_plate_distance(sim, arm: str) -> float:
    """Return Euclidean distance from an arm gripper to the plate in meters."""
    gripper_position = np.asarray(
        sim.get_gripper_position(arm),
        dtype=float,
    )
    plate_position = get_plate_position(sim)
    return float(np.linalg.norm(gripper_position - plate_position))
