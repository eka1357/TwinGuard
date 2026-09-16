"""Smoke test for the Jacobian-based reachability probe."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.jacobian_reach import move_gripper_toward
from simulation.simulator import TwinGuardSim


def main() -> None:
    sim = TwinGuardSim()

    try:
        sim.reset()

        initial_gripper_position = np.asarray(
            sim.get_gripper_position("left_arm"),
            dtype=float,
        )
        plate_position = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )
        target = plate_position.copy()
        target[2] += 0.10
        initial_distance = float(
            np.linalg.norm(initial_gripper_position - target)
        )

        print(f"Initial gripper position: {initial_gripper_position}")
        print(f"Plate position: {plate_position}")
        print(f"Target position: {target}")
        print(f"Initial distance: {initial_distance:.4f} m")

        final_distance = move_gripper_toward(
            sim,
            "left_arm",
            target,
            max_iterations=300,
            position_tolerance=0.03,
            step_scale=0.10,
        )

        final_gripper_position = np.asarray(
            sim.get_gripper_position("left_arm"),
            dtype=float,
        )
        print(f"Final gripper position: {final_gripper_position}")
        print(f"Final distance: {final_distance:.4f} m")

        assert sim.is_stable()
        print("[SUCCESS] Real Jacobian reach probe completed.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
