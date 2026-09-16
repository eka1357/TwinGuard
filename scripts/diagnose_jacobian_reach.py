"""Diagnostic copy of the Jacobian reach controller with periodic reporting."""

import sys
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.pose_validation import JOINT_LIMITS
from simulation.simulator import TwinGuardSim


NON_GRIPPER_JOINTS = (
    "joint_base_yaw",
    "joint_shoulder_pitch",
    "joint_elbow_pitch",
    "joint_wrist_pitch",
    "joint_wrist_roll",
)


def _near_limit_joints(positions: Mapping[str, float]) -> list[str]:
    margin_degrees = 5.0
    return [
        joint
        for joint in NON_GRIPPER_JOINTS
        if (
            positions[joint] <= JOINT_LIMITS[joint][0] + margin_degrees
            or positions[joint] >= JOINT_LIMITS[joint][1] - margin_degrees
        )
    ]


def run_diagnostic(
    sim,
    arm: str,
    target: np.ndarray,
    *,
    max_iterations: int = 300,
    position_tolerance: float = 0.03,
    step_scale: float = 0.10,
) -> float:
    """Run a diagnostic copy of the Jacobian reach algorithm."""
    site_name = f"{arm}_gripper_site"
    site_id = mujoco.mj_name2id(
        sim.model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )
    if site_id < 0:
        raise ValueError(f"Site '{site_name}' not found in the model.")

    joint_dof_addresses: Mapping[str, int] = {}
    for canonical_joint in NON_GRIPPER_JOINTS:
        joint_name = f"{arm}_{canonical_joint}"
        joint_id = mujoco.mj_name2id(
            sim.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise ValueError(f"Joint '{joint_name}' not found in the model.")
        joint_dof_addresses[canonical_joint] = int(
            sim.model.jnt_dofadr[joint_id]
        )

    jacp = np.zeros((3, sim.model.nv))
    jacr = np.zeros((3, sim.model.nv))
    damping = 0.05
    distance = float(
        np.linalg.norm(np.asarray(sim.data.site_xpos[site_id]) - target)
    )

    for iteration in range(1, max_iterations + 1):
        if distance <= position_tolerance:
            break

        mujoco.mj_jacSite(sim.model, sim.data, jacp, jacr, site_id)
        dof_addresses = list(joint_dof_addresses.values())
        arm_jacobian = jacp[:, dof_addresses]
        position_error = target - np.asarray(
            sim.data.site_xpos[site_id],
            dtype=float,
        )
        system = arm_jacobian @ arm_jacobian.T + damping**2 * np.eye(3)
        dq = arm_jacobian.T @ np.linalg.solve(system, position_error)

        current_positions = sim.get_arm_joint_positions(arm)
        targets = {}
        for index, canonical_joint in enumerate(NON_GRIPPER_JOINTS):
            lower, upper = JOINT_LIMITS[canonical_joint]
            updated_target = (
                current_positions[canonical_joint]
                + step_scale * float(np.degrees(dq[index]))
            )
            targets[canonical_joint] = float(
                np.clip(updated_target, lower, upper)
            )

        sim.set_arm_joint_targets(arm, targets)
        sim.step(1)

        if not sim.is_stable():
            raise RuntimeError("Simulation became unstable during diagnostic.")

        distance = float(
            np.linalg.norm(np.asarray(sim.data.site_xpos[site_id]) - target)
        )

        if iteration % 25 == 0:
            positions = sim.get_arm_joint_positions(arm)
            near_limits = _near_limit_joints(positions)
            print(f"Iteration: {iteration}")
            print(f"Gripper: {sim.get_gripper_position(arm)}")
            print(f"Distance: {distance:.4f} m")
            print(
                "Non-gripper joints (degrees): "
                f"{ {joint: positions[joint] for joint in NON_GRIPPER_JOINTS} }"
            )
            print(f"Any joint at/near limit: {bool(near_limits)}")
            print(f"Joints at/near limit: {near_limits}")

    return distance


def main() -> None:
    sim = TwinGuardSim()

    try:
        sim.reset()

        plate_position = np.asarray(
            sim.get_joint_positions()["plate_joint"][:3],
            dtype=float,
        )
        target = plate_position.copy()
        target[2] += 0.10

        initial_gripper = np.asarray(
            sim.get_gripper_position("left_arm"),
            dtype=float,
        )
        initial_distance = float(np.linalg.norm(initial_gripper - target))
        print(f"Initial gripper position: {initial_gripper}")
        print(f"Initial distance: {initial_distance:.4f} m")

        final_distance = run_diagnostic(sim, "left_arm", target)
        print(f"Final distance: {final_distance:.4f} m")
        print("[SUCCESS] Jacobian diagnostic completed.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
