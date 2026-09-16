"""Diagnostic probe for gripper-plate contacts in the real simulator."""

import sys
from pathlib import Path

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.jacobian_reach import move_gripper_toward
from simulation.simulator import TwinGuardSim


def print_body_position(sim, body_name: str) -> None:
    body_id = mujoco.mj_name2id(
        sim.model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in the model.")
    print(f"{body_name} position: {sim.data.xpos[body_id].copy()}")


def print_contacts(sim, label: str) -> None:
    print(f"{label} contacts:")
    for index in range(sim.data.ncon):
        contact = sim.data.contact[index]
        geom1_name = sim.model.geom(contact.geom1).name
        geom2_name = sim.model.geom(contact.geom2).name
        print(
            f"{index}: {geom1_name} | {geom2_name} | "
            f"dist={contact.dist:.6f}"
        )


def main() -> None:
    sim = TwinGuardSim()

    try:
        sim.reset()

        initial_plate = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )
        print(f"Initial plate position: {initial_plate}")
        print(
            "left_arm_gripper_site position: "
            f"{sim.get_site_position('left_arm_gripper_site')}"
        )
        print_body_position(sim, "left_arm_gripper_finger_left")
        print_body_position(sim, "left_arm_gripper_finger_right")

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
        print_contacts(sim, "Contacts before closing")

        sim.close_gripper("left_arm")
        sim.step(200)

        plate_after_closing = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )
        print(f"Plate position after closing: {plate_after_closing}")
        print_contacts(sim, "Contacts after closing")

        print("[DONE] Gripper contact diagnostic completed.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
