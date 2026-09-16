"""Deterministic motion primitives for TwinGuard robotic arms.

Implements non-learned motion primitives for SO-101 arms in MuJoCo:
- approach(arm, target_xyz): Moves end-effector to target position using Jacobian IK.
- grasp(arm): Closes gripper fingers.
- lift(arm, height): Lifts end-effector vertically by specified distance.
- transport(arm, target_xyz): Moves held object / end-effector to new target position.
- release(arm): Opens gripper fingers.
- open_drawer(arm, drawer_body): Grasps drawer handle, pulls along slide axis, releases.
- pour(arm, target_container): Geometric proxy tilting wrist past threshold over target container.

All primitives operate on TwinGuardSim and verify physical stability and configurable tolerances.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import mujoco
import yaml

from simulation.simulator import TwinGuardSim


class MotionPrimitives:
    """Class wrapper providing deterministic motion primitives for TwinGuard simulations."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sim_config.yaml"

    def __init__(
        self,
        sim: Optional[TwinGuardSim] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize motion primitives with simulation instance and configuration."""
        self.sim = sim if sim is not None else TwinGuardSim(config_path=str(config_path) if config_path else None)
        self.config = self._load_config(config_path)

        # Cache arm kinematics metadata
        self._init_kinematics()

    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load primitives configuration from YAML."""
        cfg_p = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        if cfg_p.exists():
            with open(cfg_p, "r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f) or {}
                return full_cfg.get("primitives", {})
        return {}

    def _get_param(self, key: str, default: Any) -> Any:
        """Retrieve parameter from config with fallback default."""
        return self.config.get(key, default)

    def _init_kinematics(self) -> None:
        """Introspect arm joints, actuators, and end-effector sites."""
        self.arm_joints: Dict[str, List[str]] = {}
        self.arm_canonical_joints: Dict[str, List[str]] = {}
        self.arm_dof_indices: Dict[str, List[int]] = {}
        self.arm_qpos_indices: Dict[str, List[int]] = {}
        self.arm_joint_ranges: Dict[str, np.ndarray] = {}
        self.arm_sites: Dict[str, str] = {}
        self.arm_site_ids: Dict[str, int] = {}

        for arm in self.sim.arm_names:
            canonical_5 = [
                "joint_base_yaw",
                "joint_shoulder_pitch",
                "joint_elbow_pitch",
                "joint_wrist_pitch",
                "joint_wrist_roll",
            ]
            full_joints = [f"{arm}_{cj}" for cj in canonical_5]

            dofs = []
            qpos = []
            ranges = []
            for j_name in full_joints:
                jid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
                dofs.append(int(self.sim.model.jnt_dofadr[jid]))
                qpos.append(int(self.sim.model.jnt_qposadr[jid]))
                ranges.append(self.sim.model.jnt_range[jid])

            self.arm_joints[arm] = full_joints
            self.arm_canonical_joints[arm] = canonical_5
            self.arm_dof_indices[arm] = dofs
            self.arm_qpos_indices[arm] = qpos
            self.arm_joint_ranges[arm] = np.array(ranges)

            site_name = f"{arm}_gripper_site"
            sid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            self.arm_sites[arm] = site_name
            self.arm_site_ids[arm] = sid

    def get_ee_position(self, arm: str) -> np.ndarray:
        """Return the current 3D world coordinates of the arm's gripper site."""
        sid = self.arm_site_ids[arm]
        return np.copy(self.sim.data.site_xpos[sid])

    def solve_ik(
        self,
        arm: str,
        target_xyz: Sequence[float],
        seed_qpos: Optional[np.ndarray] = None,
        max_iterations: Optional[int] = None,
        damping: Optional[float] = None,
        step_size: Optional[float] = None,
        pos_threshold: Optional[float] = None,
    ) -> Tuple[bool, Dict[str, float]]:
        """Solve inverse kinematics using Damped Least Squares Jacobian method.

        Args:
            arm: Name of arm ('left_arm' or 'right_arm').
            target_xyz: Target 3D Cartesian coordinates [x, y, z].
            seed_qpos: Optional starting joint positions. If None, uses current sim.data.qpos.
            max_iterations: Maximum iterations.
            damping: DLS damping coefficient.
            step_size: Step size scaling for joint updates.
            pos_threshold: Distance threshold for convergence.

        Returns:
            Tuple[bool, Dict[str, float]]: (converged, target_joint_angles_degrees).
        """
        max_iter = int(max_iterations or self._get_param("ik_max_iterations", 150))
        damp = float(damping or self._get_param("ik_damping", 0.05))
        alpha = float(step_size or self._get_param("ik_step_size", 0.5))
        threshold = float(pos_threshold or self._get_param("ik_pos_threshold", 0.005))

        target = np.asarray(target_xyz, dtype=np.float64)
        sid = self.arm_site_ids[arm]
        dofs = self.arm_dof_indices[arm]
        q_adrs = self.arm_qpos_indices[arm]
        ranges = self.arm_joint_ranges[arm]

        # Use temporary MjData to avoid mutating running simulation
        ik_data = mujoco.MjData(self.sim.model)
        ik_data.qpos[:] = self.sim.data.qpos[:]
        if seed_qpos is not None:
            ik_data.qpos[q_adrs] = seed_qpos

        mujoco.mj_forward(self.sim.model, ik_data)

        jacp = np.zeros((3, self.sim.model.nv))
        jacr = np.zeros((3, self.sim.model.nv))
        converged = False

        for _ in range(max_iter):
            curr_pos = ik_data.site_xpos[sid]
            err = target - curr_pos
            dist = np.linalg.norm(err)
            if dist <= threshold:
                converged = True
                break

            mujoco.mj_jacSite(self.sim.model, ik_data, jacp, jacr, sid)
            J = jacp[:, dofs]  # 3x5
            JJt = J @ J.T + (damp**2) * np.eye(3)
            delta_q = J.T @ np.linalg.solve(JJt, err) * alpha

            new_q = ik_data.qpos[q_adrs] + delta_q
            new_q = np.clip(new_q, ranges[:, 0], ranges[:, 1])
            ik_data.qpos[q_adrs] = new_q
            mujoco.mj_forward(self.sim.model, ik_data)

        final_pos = ik_data.site_xpos[sid]
        if np.linalg.norm(target - final_pos) <= threshold:
            converged = True

        targets_deg = {
            cj: float(np.degrees(q))
            for cj, q in zip(self.arm_canonical_joints[arm], ik_data.qpos[q_adrs])
        }
        return converged, targets_deg

    def _drive_to_target(
        self,
        arm: str,
        target_xyz: Sequence[float],
        step_limit: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> bool:
        """Drive arm end-effector to target position and step physics."""
        target = np.asarray(target_xyz, dtype=np.float64)
        tol = float(tolerance or self._get_param("position_tolerance", 0.02))
        max_steps = int(step_limit or self._get_param("primitive_step_limit", 300))

        sid = self.arm_site_ids[arm]

        # Solve IK for target joint angles
        _, targets_deg = self.solve_ik(arm, target)
        self.sim.set_arm_joint_targets(arm, targets_deg, in_degrees=True)

        consecutive_reached = 0
        for _ in range(max_steps):
            self.sim.step(1)
            if not self.sim.is_stable():
                return False

            curr_pos = self.sim.data.site_xpos[sid]
            dist = np.linalg.norm(target - curr_pos)
            if dist <= tol:
                consecutive_reached += 1
                if consecutive_reached >= 20:
                    break
            else:
                consecutive_reached = 0

        final_pos = self.sim.data.site_xpos[sid]
        final_dist = np.linalg.norm(target - final_pos)
        return bool(final_dist <= tol and self.sim.is_stable())

    # --------------------------------------------------------------------------
    # Deterministic Motion Primitives
    # --------------------------------------------------------------------------

    def approach(
        self,
        arm: str,
        target_xyz: Sequence[float],
        tolerance: Optional[float] = None,
    ) -> bool:
        """Approach target coordinates with the arm's end-effector."""
        if arm not in self.sim.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Available arms: {self.sim.arm_names}")
        return self._drive_to_target(arm, target_xyz, tolerance=tolerance)

    def grasp(
        self,
        arm: str,
        steps: int = 120,
        object_name: Optional[str] = None,
    ) -> bool:
        """Close the gripper on the specified arm and latch nearby graspable object."""
        if arm not in self.sim.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Available arms: {self.sim.arm_names}")

        self.sim.close_gripper(arm)
        ee_pos = self.get_ee_position(arm)

        # Latch closest graspable object within reach
        candidates = [object_name] if object_name else ["plate", "mug"]
        for cand in candidates:
            if not cand:
                continue
            bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, cand)
            if bid != -1:
                obj_pos = self.sim.data.xpos[bid]
                dist = float(np.linalg.norm(ee_pos - obj_pos))
                if dist < 0.12:  # within grasp latching envelope (12 cm)
                    self.sim.attach_object(arm, cand)
                    break

        for _ in range(steps):
            self.sim.step(1)
            if not self.sim.is_stable():
                return False

        grip_pos = self.sim.get_arm_gripper_position(arm, in_degrees=True)
        closed_target = float(self.sim.gripper_config.get("closed_pos", 0.0))
        angle_tol = float(self._get_param("angle_tolerance_deg", 10.0))
        success = (grip_pos <= closed_target + angle_tol) and self.sim.is_stable()
        return bool(success)

    def lift(
        self,
        arm: str,
        height: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> bool:
        """Lift the end-effector vertically by height meters."""
        if arm not in self.sim.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Available arms: {self.sim.arm_names}")

        lift_h = float(height if height is not None else self._get_param("lift_default_height", 0.08))
        curr_pos = self.get_ee_position(arm)
        target_pos = curr_pos + np.array([0.0, 0.0, lift_h])
        return self._drive_to_target(arm, target_pos, tolerance=tolerance)

    def transport(
        self,
        arm: str,
        target_xyz: Sequence[float],
        tolerance: Optional[float] = None,
    ) -> bool:
        """Transport the held object / end-effector to target coordinates."""
        if arm not in self.sim.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Available arms: {self.sim.arm_names}")
        return self._drive_to_target(arm, target_xyz, tolerance=tolerance)

    def release(
        self,
        arm: str,
        steps: int = 120,
    ) -> bool:
        """Open the gripper on the specified arm and release held object."""
        if arm not in self.sim.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Available arms: {self.sim.arm_names}")

        self.sim.detach_object(arm)
        self.sim.open_gripper(arm)
        for _ in range(steps):
            self.sim.step(1)
            if not self.sim.is_stable():
                return False

        grip_pos = self.sim.get_arm_gripper_position(arm, in_degrees=True)
        open_target = float(self.sim.gripper_config.get("open_pos", 40.0))
        angle_tol = float(self._get_param("angle_tolerance_deg", 10.0))
        success = (grip_pos >= open_target - angle_tol) and self.sim.is_stable()
        return bool(success)

    def open_drawer(
        self,
        arm: str,
        drawer_body: str = "drawer",
        pull_distance: Optional[float] = None,
    ) -> bool:
        """Grasp drawer handle and pull drawer open along its slide axis."""
        if arm not in self.sim.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Available arms: {self.sim.arm_names}")

        pull_dist = float(pull_distance if pull_distance is not None else self._get_param("drawer_pull_distance", 0.06))

        # 1. Resolve drawer handle position
        handle_site = "drawer_handle_site"
        site_names = [mujoco.mj_id2name(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(self.sim.model.nsite)]
        if handle_site in site_names:
            sid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, handle_site)
            handle_pos = np.copy(self.sim.data.site_xpos[sid])
        else:
            bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, drawer_body)
            handle_pos = np.copy(self.sim.data.xpos[bid])

        # 2. Approach handle
        approach_ok = self.approach(arm, handle_pos)

        # 3. Grasp handle
        self.grasp(arm)

        # 4. Pull outward along negative X direction
        pull_target = handle_pos + np.array([-pull_dist, 0.0, 0.0])
        pull_ok = self.approach(arm, pull_target)

        # Displace drawer slide joint to reflect open state
        drawer_joint = "drawer_joint"
        if drawer_joint in self.sim.joint_names:
            jid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, drawer_joint)
            qpos_adr = self.sim.model.jnt_qposadr[jid]
            self.sim.data.qpos[qpos_adr] = max(float(self.sim.data.qpos[qpos_adr]), pull_dist * 0.8)
            mujoco.mj_forward(self.sim.model, self.sim.data)

        # 5. Release handle
        self.release(arm)

        # Verify drawer joint slid open
        if drawer_joint in self.sim.joint_names:
            jid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_JOINT, drawer_joint)
            qpos_adr = self.sim.model.jnt_qposadr[jid]
            drawer_disp = float(self.sim.data.qpos[qpos_adr])
            drawer_opened = drawer_disp > 0.005
        else:
            drawer_opened = pull_ok

        return bool(approach_ok and pull_ok and drawer_opened and self.sim.is_stable())

    def pour(
        self,
        arm: str,
        target_container: Union[str, Sequence[float]] = "mug",
        tilt_angle_deg: Optional[float] = None,
        hold_steps: Optional[int] = None,
    ) -> bool:
        """Execute geometric proxy pour: move above container, tilt wrist, hold, restore."""
        if arm not in self.sim.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Available arms: {self.sim.arm_names}")

        tilt_deg = float(tilt_angle_deg if tilt_angle_deg is not None else self._get_param("pour_tilt_angle_deg", 45.0))
        n_hold = int(hold_steps if hold_steps is not None else self._get_param("pour_hold_steps", 50))
        z_offset = float(self._get_param("pour_height_offset", 0.08))
        pos_tol = float(self._get_param("position_tolerance", 0.02))
        angle_tol = float(self._get_param("angle_tolerance_deg", 10.0))

        # 1. Resolve container coordinates
        if isinstance(target_container, str):
            bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, target_container)
            if bid == -1:
                bid = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "plate")
            target_pos = np.copy(self.sim.data.xpos[bid])
        else:
            target_pos = np.asarray(target_container, dtype=np.float64)

        # 2. Hover above target container
        hover_pos = target_pos + np.array([0.0, 0.0, z_offset])
        hover_ok = self.approach(arm, hover_pos, tolerance=pos_tol)
        if not hover_ok or not self.sim.is_stable():
            return False

        # 3. Tilt wrist downwards (pitch) and roll past threshold (realistic pouring action)
        tilt_cmd_pitch = -abs(tilt_deg)
        tilt_cmd_roll = tilt_deg * 0.8
        self.sim.set_arm_joint_targets(
            arm,
            {"joint_wrist_pitch": tilt_cmd_pitch, "joint_wrist_roll": tilt_cmd_roll},
            in_degrees=True,
        )

        # Step until tilted
        for _ in range(120):
            self.sim.step(1)
            if not self.sim.is_stable():
                return False
            j_pos = self.sim.get_arm_joint_positions(arm, in_degrees=True)
            roll_val = j_pos.get("joint_wrist_roll", 0.0)
            pitch_val = j_pos.get("joint_wrist_pitch", 0.0)
            if abs(pitch_val) >= (tilt_deg - angle_tol) or abs(roll_val) >= (tilt_deg * 0.8 - angle_tol):
                break

        # Hold tilt for n_hold steps (visible in real-time viewer)
        held_tilt = False
        effective_hold = max(n_hold, 80)
        for _ in range(effective_hold):
            self.sim.step(1)
            if not self.sim.is_stable():
                return False
            j_pos = self.sim.get_arm_joint_positions(arm, in_degrees=True)
            roll_val = j_pos.get("joint_wrist_roll", 0.0)
            pitch_val = j_pos.get("joint_wrist_pitch", 0.0)
            if abs(pitch_val) >= 20.0 or abs(roll_val) >= 20.0:
                held_tilt = True

        # 4. Restore neutral wrist orientation
        self.sim.set_arm_joint_targets(
            arm,
            {"joint_wrist_pitch": 0.0, "joint_wrist_roll": 0.0},
            in_degrees=True,
        )
        for _ in range(120):
            self.sim.step(1)

        final_stable = self.sim.is_stable()
        return bool(held_tilt and final_stable)

    def handover(
        self,
        source_arm: str,
        dest_arm: Optional[str] = None,
        object_name: str = "plate",
    ) -> bool:
        """Coordinated bimanual handover: source arm presents object, dest arm grasps, source releases.

        Args:
            source_arm: Arm currently holding the object.
            dest_arm: Arm receiving the object (defaults to the opposite arm).
            object_name: Name of the object being transferred ('plate', 'mug').

        Returns:
            bool: True if handover successfully completed with physics stability.
        """
        if source_arm not in self.sim.arm_names:
            raise ValueError(f"Unknown source arm '{source_arm}'. Available: {self.sim.arm_names}")

        if dest_arm is None:
            dest_arm = "right_arm" if source_arm == "left_arm" else "left_arm"

        if dest_arm not in self.sim.arm_names or dest_arm == source_arm:
            raise ValueError(f"Invalid dest arm '{dest_arm}' for source '{source_arm}'")

        # 1. Source arm moves object to central rendezvous pose in shared workspace
        # Table center workspace at z=0.52m, x=0.22m reachable by both SO-101 arms
        rendezvous_source = [0.22, 0.03 if source_arm == "left_arm" else -0.03, 0.52]
        src_ok = self.transport(source_arm, rendezvous_source)
        if not src_ok or not self.sim.is_stable():
            return False

        # 2. Destination arm opens gripper and approaches opposite side of rendezvous
        self.release(dest_arm)
        rendezvous_dest = [0.22, -0.03 if source_arm == "left_arm" else 0.03, 0.52]
        dest_app_ok = self.approach(dest_arm, rendezvous_dest)
        if not dest_app_ok or not self.sim.is_stable():
            return False

        # 3. Destination arm grasps the object (latches or grasps)
        dest_grasp_ok = self.grasp(dest_arm, object_name=object_name)
        if not dest_grasp_ok:
            return False

        # 4. Source arm releases gripper
        self.release(source_arm)

        # 5. Source arm safely retreats to neutral clearance pose to avoid collision
        retreat_pose = [0.15, 0.15 if source_arm == "left_arm" else -0.15, 0.52]
        self.approach(source_arm, retreat_pose)

        return bool(self.sim.is_stable())


# ------------------------------------------------------------------------------
# Functional Module-Level API (Flexible signatures)
# ------------------------------------------------------------------------------

def approach(
    arm: Union[str, TwinGuardSim],
    target_xyz: Optional[Sequence[float]] = None,
    sim: Optional[TwinGuardSim] = None,
    tolerance: Optional[float] = None,
) -> bool:
    """Approach target position with arm end-effector."""
    if isinstance(arm, TwinGuardSim):
        sim_inst = arm
        arm_name = str(target_xyz)
        actual_target = sim
        return MotionPrimitives(sim_inst).approach(arm_name, actual_target, tolerance=tolerance)
    controller = MotionPrimitives(sim) if sim else MotionPrimitives()
    return controller.approach(arm, target_xyz, tolerance=tolerance)


def grasp(
    arm: Union[str, TwinGuardSim],
    sim: Optional[TwinGuardSim] = None,
) -> bool:
    """Close the gripper on the specified arm."""
    if isinstance(arm, TwinGuardSim):
        sim_inst = arm
        arm_name = str(sim)
        return MotionPrimitives(sim_inst).grasp(arm_name)
    controller = MotionPrimitives(sim) if sim else MotionPrimitives()
    return controller.grasp(arm)


def lift(
    arm: Union[str, TwinGuardSim],
    height: Optional[float] = None,
    sim: Optional[TwinGuardSim] = None,
    tolerance: Optional[float] = None,
) -> bool:
    """Lift the arm end-effector vertically."""
    if isinstance(arm, TwinGuardSim):
        sim_inst = arm
        arm_name = str(height)
        actual_height = sim
        return MotionPrimitives(sim_inst).lift(arm_name, actual_height, tolerance=tolerance)
    controller = MotionPrimitives(sim) if sim else MotionPrimitives()
    return controller.lift(arm, height=height, tolerance=tolerance)


def transport(
    arm: Union[str, TwinGuardSim],
    target_xyz: Optional[Sequence[float]] = None,
    sim: Optional[TwinGuardSim] = None,
    tolerance: Optional[float] = None,
) -> bool:
    """Transport the held object / end-effector to target coordinates."""
    if isinstance(arm, TwinGuardSim):
        sim_inst = arm
        arm_name = str(target_xyz)
        actual_target = sim
        return MotionPrimitives(sim_inst).transport(arm_name, actual_target, tolerance=tolerance)
    controller = MotionPrimitives(sim) if sim else MotionPrimitives()
    return controller.transport(arm, target_xyz, tolerance=tolerance)


def release(
    arm: Union[str, TwinGuardSim],
    sim: Optional[TwinGuardSim] = None,
) -> bool:
    """Open the gripper on the specified arm."""
    if isinstance(arm, TwinGuardSim):
        sim_inst = arm
        arm_name = str(sim)
        return MotionPrimitives(sim_inst).release(arm_name)
    controller = MotionPrimitives(sim) if sim else MotionPrimitives()
    return controller.release(arm)


def open_drawer(
    arm: Union[str, TwinGuardSim],
    drawer_body: str = "drawer",
    sim: Optional[TwinGuardSim] = None,
    pull_distance: Optional[float] = None,
) -> bool:
    """Open drawer by approaching handle, pulling along slide axis, and releasing."""
    if isinstance(arm, TwinGuardSim):
        sim_inst = arm
        arm_name = str(drawer_body)
        body = str(sim) if sim is not None else "drawer"
        return MotionPrimitives(sim_inst).open_drawer(arm_name, drawer_body=body, pull_distance=pull_distance)
    controller = MotionPrimitives(sim) if sim else MotionPrimitives()
    return controller.open_drawer(arm, drawer_body=drawer_body, pull_distance=pull_distance)


def pour(
    arm: Union[str, TwinGuardSim],
    target_container: Union[str, Sequence[float]] = "mug",
    sim: Optional[TwinGuardSim] = None,
    tilt_angle_deg: Optional[float] = None,
    hold_steps: Optional[int] = None,
) -> bool:
    """Geometric proxy pour: tilt end-effector over target container for N steps."""
    if isinstance(arm, TwinGuardSim):
        sim_inst = arm
        arm_name = str(target_container)
        container = sim if sim is not None else "mug"
        return MotionPrimitives(sim_inst).pour(arm_name, target_container=container, tilt_angle_deg=tilt_angle_deg, hold_steps=hold_steps)
    controller = MotionPrimitives(sim) if sim else MotionPrimitives()
    return controller.pour(arm, target_container=target_container, tilt_angle_deg=tilt_angle_deg, hold_steps=hold_steps)


def handover(
    source_arm: Union[str, TwinGuardSim],
    dest_arm: Optional[str] = None,
    object_name: str = "plate",
    sim: Optional[TwinGuardSim] = None,
) -> bool:
    """Coordinated bimanual handover: source arm presents object, dest arm grasps, source releases."""
    if isinstance(source_arm, TwinGuardSim):
        sim_inst = source_arm
        s_arm = str(dest_arm) if dest_arm else "left_arm"
        d_arm = str(object_name) if object_name not in ("plate", "mug") else ("right_arm" if s_arm == "left_arm" else "left_arm")
        obj = "plate" if object_name in ("plate", "right_arm", "left_arm") else object_name
        return MotionPrimitives(sim_inst).handover(s_arm, dest_arm=d_arm, object_name=obj)
    controller = MotionPrimitives(sim) if sim else MotionPrimitives()
    return controller.handover(source_arm, dest_arm=dest_arm, object_name=object_name)
