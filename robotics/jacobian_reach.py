from typing import Mapping

import numpy as np
import mujoco

from robotics.pose_validation import JOINT_LIMITS


_NON_GRIPPER_JOINTS = (
    "joint_base_yaw",
    "joint_shoulder_pitch",
    "joint_elbow_pitch",
    "joint_wrist_pitch",
    "joint_wrist_roll",
)


def move_gripper_toward(
    sim,
    arm: str,
    target_position: np.ndarray,
    *,
    max_iterations: int = 100,
    position_tolerance: float = 0.02,
    step_scale: float = 0.25,
    steps_per_update: int = 10,
    damping: float = 0.05,
) -> float:
    """Move an arm's gripper site toward an XYZ target using a position Jacobian.

    Return the final Euclidean distance to the target in meters.
    This is only a reachability probe. It must not close the gripper,
    attach objects, or implement pick/place behavior. ``steps_per_update``
    allows the position actuators to settle after each target update.
    """
    if arm not in sim.arm_names:
        raise ValueError(f"Unknown arm '{arm}'. Configured arms: {sim.arm_names}")

    target = np.asarray(target_position, dtype=float)
    if target.shape != (3,):
        raise ValueError("target_position must have shape (3,).")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be greater than zero.")
    if position_tolerance <= 0:
        raise ValueError("position_tolerance must be greater than zero.")
    if step_scale <= 0:
        raise ValueError("step_scale must be greater than zero.")
    if steps_per_update <= 0:
        raise ValueError("steps_per_update must be greater than zero.")
    if damping <= 0:
        raise ValueError("damping must be greater than zero.")

    site_name = f"{arm}_gripper_site"
    site_id = mujoco.mj_name2id(
        sim.model,
        mujoco.mjtObj.mjOBJ_SITE,
        site_name,
    )
    if site_id < 0:
        raise ValueError(f"Site '{site_name}' not found in the model.")

    joint_dof_addresses: Mapping[str, int] = {}
    for canonical_joint in _NON_GRIPPER_JOINTS:
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

    def current_distance() -> float:
        position = np.asarray(sim.data.site_xpos[site_id], dtype=float)
        return float(np.linalg.norm(position - target))

    distance = current_distance()
    for _ in range(max_iterations):
        if distance <= position_tolerance:
            return distance

        mujoco.mj_jacSite(sim.model, sim.data, jacp, jacr, site_id)
        dof_addresses = list(joint_dof_addresses.values())
        arm_jacobian = jacp[:, dof_addresses]
        position_error = target - np.asarray(
            sim.data.site_xpos[site_id],
            dtype=float,
        )
        system = (
            arm_jacobian @ arm_jacobian.T
            + damping**2 * np.eye(3)
        )
        dq = arm_jacobian.T @ np.linalg.solve(system, position_error)

        current_positions = sim.get_arm_joint_positions(arm)
        targets = {}
        for index, canonical_joint in enumerate(_NON_GRIPPER_JOINTS):
            lower, upper = JOINT_LIMITS[canonical_joint]
            updated_target = (
                current_positions[canonical_joint]
                + step_scale * float(np.degrees(dq[index]))
            )
            targets[canonical_joint] = float(
                np.clip(updated_target, lower, upper)
            )

        sim.set_arm_joint_targets(arm, targets)
        # Let the position actuators settle while checking every physics step.
        for _ in range(steps_per_update):
            sim.step(1)

            if not sim.is_stable():
                raise RuntimeError("Simulation became unstable during Jacobian reach.")

        distance = current_distance()

    return distance
