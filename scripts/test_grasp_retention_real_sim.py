"""Controlled grasp-retention diagnostic for the real simulator."""

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.jacobian_reach import move_gripper_toward
from simulation.simulator import TwinGuardSim


CONTACT_GEOMS = {
    "plate_geom",
    "plate_rim",
    "left_arm_finger_left_geom",
    "left_arm_finger_right_geom",
}
TABLE_TOP_Z = 0.42
MIN_RETAINED_HEIGHT = 0.05
MAX_SETTLING_DROP = 0.03
MAX_HORIZONTAL_DISTANCE = 0.15


def print_relevant_contacts(sim, label: str) -> None:
    counts = {name: 0 for name in sorted(CONTACT_GEOMS)}
    total = 0

    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        geom1_name = sim.model.geom(contact.geom1).name
        geom2_name = sim.model.geom(contact.geom2).name
        involved = {geom1_name, geom2_name} & CONTACT_GEOMS
        if not involved:
            continue

        total += 1
        for geom_name in involved:
            counts[geom_name] += 1

    print(f"{label} relevant contact count: {total}")
    for geom_name, count in counts.items():
        print(f"{label} {geom_name}: {count}")


def horizontal_distance(sim, plate_position: np.ndarray) -> float:
    gripper_position = np.asarray(
        sim.get_site_position("left_arm_gripper_site"),
        dtype=float,
    )
    return float(np.linalg.norm(plate_position[:2] - gripper_position[:2]))


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

        sim.open_gripper("left_arm")
        sim.step(50)

        plate_before_closing = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )
        print(f"Plate position before closing: {plate_before_closing}")

        sim.close_gripper("left_arm")
        sim.step(50)
        plate_after_50 = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )
        print(f"Plate position after 50 closing steps: {plate_after_50}")
        print_relevant_contacts(sim, "After 50 closing steps")

        sim.step(200)
        plate_after_settling = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )
        print(
            "Plate position after 200 settling steps: "
            f"{plate_after_settling}"
        )
        print_relevant_contacts(sim, "After 200 settling steps")

        horizontal_after_50 = horizontal_distance(sim, plate_after_50)
        horizontal_after_settling = horizontal_distance(
            sim,
            plate_after_settling,
        )
        settling_drop = plate_after_50[2] - plate_after_settling[2]
        retained_height = plate_after_settling[2] - TABLE_TOP_Z

        print(f"Plate z before closing: {plate_before_closing[2]:.4f} m")
        print(f"Plate z after 50 steps: {plate_after_50[2]:.4f} m")
        print(
            "Plate z after 200 settling steps: "
            f"{plate_after_settling[2]:.4f} m"
        )
        print(f"Horizontal distance after 50 steps: {horizontal_after_50:.4f} m")
        print(
            "Horizontal distance after settling: "
            f"{horizontal_after_settling:.4f} m"
        )
        print(f"Settling z drop: {settling_drop:.4f} m")
        print(f"Retained height above table: {retained_height:.4f} m")

        retention_passed = (
            sim.is_stable()
            and retained_height >= MIN_RETAINED_HEIGHT
            and settling_drop <= MAX_SETTLING_DROP
            and horizontal_after_settling <= MAX_HORIZONTAL_DISTANCE
        )

        if retention_passed:
            print("[SUCCESS] Stable grasp retention demonstrated.")
        else:
            print("[RESULT] Stable grasp retention was not demonstrated.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
