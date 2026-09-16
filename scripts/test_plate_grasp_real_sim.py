"""Standalone real-simulator plate grasp experiment."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.grasp_checks import is_gripper_near_plate
from robotics.jacobian_reach import move_gripper_toward
from robotics.motion import move_arm_to_targets
from robotics.reachability import gripper_to_plate_distance
from simulation.simulator import TwinGuardSim


LIFT_SHOULDER_DELTA_DEGREES = -5.0
TABLE_SURFACE_Z = 0.42
PROXIMITY_THRESHOLD = 0.08
ELEVATION_MARGIN = 0.01


def main() -> None:
    sim = TwinGuardSim()

    try:
        sim.reset()

        initial_plate = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )
        print(f"Initial plate position: {initial_plate}")

        target = initial_plate.copy()
        target[2] += 0.04

        sim.open_gripper("left_arm")
        move_gripper_toward(
            sim,
            "left_arm",
            target,
            max_iterations=300,
            position_tolerance=0.05,
            step_scale=0.10,
            steps_per_update=10,
            damping=0.05,
        )

        final_gripper = np.asarray(
            sim.get_gripper_position("left_arm"),
            dtype=float,
        )
        distance_to_plate = gripper_to_plate_distance(sim, "left_arm")
        print(f"Final gripper position: {final_gripper}")
        print(f"Distance to plate: {distance_to_plate:.4f} m")

        if not is_gripper_near_plate(
            sim,
            "left_arm",
            max_distance=PROXIMITY_THRESHOLD,
        ):
            print("[RESULT] Gripper did not reach the grasp region.")
            return

        sim.close_gripper("left_arm")
        sim.step(100)
        plate_after_closing = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )

        current_positions = sim.get_arm_joint_positions("left_arm")
        lift_targets = {
            joint: current_positions[joint]
            for joint in (
                "joint_base_yaw",
                "joint_shoulder_pitch",
                "joint_elbow_pitch",
                "joint_wrist_pitch",
                "joint_wrist_roll",
            )
        }
        lift_targets["joint_shoulder_pitch"] += LIFT_SHOULDER_DELTA_DEGREES
        move_arm_to_targets(sim, "left_arm", lift_targets, steps=100)
        sim.step(100)

        final_plate = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )
        meaningfully_elevated = (
            final_plate[2] > TABLE_SURFACE_Z + ELEVATION_MARGIN
            and final_plate[2] > plate_after_closing[2] + ELEVATION_MARGIN
        )
        print(f"Initial plate z: {initial_plate[2]:.4f} m")
        print(f"Plate z after closing: {plate_after_closing[2]:.4f} m")
        print(f"Final plate z: {final_plate[2]:.4f} m")
        print(
            "Final plate meaningfully above table surface: "
            f"{meaningfully_elevated}"
        )

        if meaningfully_elevated:
            print("[RESULT] Plate rose and remained elevated.")
        else:
            print("[RESULT] Grasp success was not demonstrated.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
