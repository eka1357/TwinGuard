"""Core MuJoCo simulation environment for TwinGuard.

Supports single-arm and bimanual dual-arm (left_arm, right_arm) SO-101 configurations
with independent arm actuation, joint telemetry, scene camera rendering, and stability checks.
All hinge joint targets and readouts operate in degrees by default (configurable via in_degrees).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import mujoco
import yaml


class TwinGuardSim:
    """TwinGuard MuJoCo simulation wrapper for bimanual SO-101 robotic arms."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "sim_config.yaml"
    DEFAULT_SCENE_PATH = PROJECT_ROOT / "simulation" / "models" / "scene.xml"

    def __init__(
        self,
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
    ):
        """Initialize the MuJoCo simulation.

        Args:
            model_path: Optional path to an MJCF XML file. If None, loaded from config.
            config_path: Optional path to YAML config. Defaults to configs/sim_config.yaml.
        """
        # Load simulation config
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        # Resolve model path from argument, config, or default
        if model_path:
            p = Path(model_path)
            self.model_path = p if p.is_absolute() else (self.PROJECT_ROOT / p)
        else:
            cfg_model = self.config.get("simulation", {}).get("model_path")
            if cfg_model:
                p = Path(cfg_model)
                self.model_path = p if p.is_absolute() else (self.PROJECT_ROOT / p)
            else:
                self.model_path = self.DEFAULT_SCENE_PATH

        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model file not found: {self.model_path}")

        # Load MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        # Introspect registered names
        self.joint_names: List[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.actuator_names: List[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        self.camera_names: List[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.model.ncam)
        ]

        # Configure arms and grippers
        self._init_arms()

        # Renderer cache keyed by (width, height)
        self._renderers: Dict[Tuple[int, int], mujoco.Renderer] = {}

    def _init_arms(self) -> None:
        """Parse arm topology from config or model structure."""
        robot_cfg = self.config.get("robot", {})
        arms_cfg = robot_cfg.get("arms", {})
        self.arm_names: List[str] = []
        self.arm_joints: Dict[str, List[str]] = {}
        self.arm_actuators: Dict[str, List[str]] = {}
        self.gripper_config: Dict[str, float] = robot_cfg.get(
            "gripper", {"open_pos": 40.0, "closed_pos": 0.0}
        )

        if arms_cfg:
            for arm_name, arm_info in arms_cfg.items():
                self.arm_names.append(arm_name)
                joints = [
                    j for j in arm_info.get("joints", [])
                    if j in self.joint_names
                ]
                actuators = [
                    a for a in arm_info.get("actuators", [])
                    if a in self.actuator_names
                ]
                self.arm_joints[arm_name] = joints
                self.arm_actuators[arm_name] = actuators
        else:
            has_left = any(j.startswith("left_arm_") for j in self.joint_names)
            has_right = any(j.startswith("right_arm_") for j in self.joint_names)
            if has_left and has_right:
                for arm_name in ["left_arm", "right_arm"]:
                    self.arm_names.append(arm_name)
                    prefix = f"{arm_name}_"
                    self.arm_joints[arm_name] = [
                        j for j in self.joint_names if j.startswith(prefix)
                    ]
                    self.arm_actuators[arm_name] = [
                        a for a in self.actuator_names if a.startswith(prefix)
                    ]
            else:
                arm_name = "arm"
                self.arm_names.append(arm_name)
                self.arm_joints[arm_name] = list(self.joint_names)
                self.arm_actuators[arm_name] = list(self.actuator_names)

    def reset(self) -> None:
        """Reset simulation state to initial zero configuration."""
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def step(self, n_steps: int = 1) -> None:
        """Advance the simulation by n physics timesteps.

        Args:
            n_steps: Number of integration steps to advance (default: 1).
        """
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def get_time(self) -> float:
        """Return current simulation time in seconds."""
        return float(self.data.time)

    def get_timestep(self) -> float:
        """Return the physics timestep (dt)."""
        return float(self.model.opt.timestep)

    def is_stable(self) -> bool:
        """Check if simulation state is physically stable (no NaNs or infinities)."""
        qpos_valid = not np.any(np.isnan(self.data.qpos)) and not np.any(np.isinf(self.data.qpos))
        qvel_valid = not np.any(np.isnan(self.data.qvel)) and not np.any(np.isinf(self.data.qvel))
        return bool(qpos_valid and qvel_valid)

    # ----------------------------------------------------------------------
    # Independent Arm Control & State API
    # ----------------------------------------------------------------------

    def set_arm_joint_targets(
        self,
        arm: str,
        targets: Dict[str, float],
        in_degrees: bool = True,
    ) -> None:
        """Apply position control targets to actuators of a specific arm.

        Args:
            arm: Name of arm (e.g. 'left_arm', 'right_arm').
            targets: Dictionary mapping actuator/joint name (canonical or full) to target angle.
            in_degrees: If True (default), targets are in degrees and converted to radians.
        """
        if arm not in self.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Configured arms: {self.arm_names}")

        actuators = self.arm_actuators[arm]

        for key, value in targets.items():
            matched_aid: Optional[int] = None

            # 1. Exact match with an actuator belonging to this arm
            if key in actuators:
                matched_aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, key)

            # 2. Prefixed actuator name: <arm>_<key>
            if matched_aid is None:
                prefixed_act = f"{arm}_{key}"
                if prefixed_act in actuators:
                    matched_aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefixed_act)

            # 3. Canonical joint name to actuator conversion
            if matched_aid is None:
                joint_suffix = key.removeprefix("joint_")
                candidate_act = f"{arm}_actuator_{joint_suffix}"
                if candidate_act in actuators:
                    matched_aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate_act)

            # 4. Short joint name
            if matched_aid is None:
                candidate_act = f"{arm}_actuator_{key}"
                if candidate_act in actuators:
                    matched_aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, candidate_act)

            if matched_aid is not None:
                ctrl_val = float(np.radians(value) if in_degrees else value)
                self.data.ctrl[matched_aid] = ctrl_val
            else:
                raise KeyError(
                    f"Could not map target '{key}' to an actuator for arm '{arm}'. "
                    f"Available actuators for {arm}: {actuators}"
                )

    def get_arm_joint_positions(self, arm: str, in_degrees: bool = True) -> Dict[str, float]:
        """Return current joint positions for a specific arm in degrees (or radians)."""
        if arm not in self.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Configured arms: {self.arm_names}")

        positions: Dict[str, float] = {}
        prefix = f"{arm}_"
        for j_name in self.arm_joints[arm]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            qpos_adr = self.model.jnt_qposadr[jid]
            rad_val = float(self.data.qpos[qpos_adr])
            val = float(np.degrees(rad_val) if in_degrees else rad_val)
            positions[j_name] = val
            if j_name.startswith(prefix):
                canonical = j_name[len(prefix):]
                positions[canonical] = val
        return positions

    def get_arm_joint_velocities(self, arm: str, in_degrees: bool = True) -> Dict[str, float]:
        """Return current joint velocities for a specific arm in deg/s (or rad/s)."""
        if arm not in self.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Configured arms: {self.arm_names}")

        velocities: Dict[str, float] = {}
        prefix = f"{arm}_"
        for j_name in self.arm_joints[arm]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j_name)
            dof_adr = self.model.jnt_dofadr[jid]
            rad_val = float(self.data.qvel[dof_adr])
            val = float(np.degrees(rad_val) if in_degrees else rad_val)
            velocities[j_name] = val
            if j_name.startswith(prefix):
                canonical = j_name[len(prefix):]
                velocities[canonical] = val
        return velocities

    def set_arm_gripper(self, arm: str, position: float, in_degrees: bool = True) -> None:
        """Command the gripper actuator for the specified arm (in degrees, range 0-40)."""
        if arm not in self.arm_names:
            raise ValueError(f"Unknown arm '{arm}'. Configured arms: {self.arm_names}")

        gripper_actuator = f"{arm}_actuator_gripper"
        if gripper_actuator not in self.actuator_names:
            gripper_actuator = "actuator_gripper"

        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator)
        ctrl_val = float(np.radians(position) if in_degrees else position)
        self.data.ctrl[aid] = ctrl_val

    def open_gripper(self, arm: str) -> None:
        """Fully open the gripper for the specified arm."""
        open_pos = float(self.gripper_config.get("open_pos", 40.0))
        self.set_arm_gripper(arm, open_pos, in_degrees=True)

    def close_gripper(self, arm: str) -> None:
        """Fully close the gripper for the specified arm."""
        closed_pos = float(self.gripper_config.get("closed_pos", 0.0))
        self.set_arm_gripper(arm, closed_pos, in_degrees=True)

    def get_arm_gripper_position(self, arm: str, in_degrees: bool = True) -> float:
        """Return the current gripper position for the specified arm in degrees."""
        positions = self.get_arm_joint_positions(arm, in_degrees=in_degrees)
        gripper_key = f"{arm}_joint_gripper"
        if gripper_key in positions:
            return positions[gripper_key]
        return positions.get("joint_gripper", 0.0)

    def get_site_position(self, site_name: str) -> np.ndarray:
        """Return the world-space position of a named MuJoCo site."""
        site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            site_name,
        )
        if site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in the model.")
        return self.data.site_xpos[site_id].copy()

    def get_gripper_position(self, arm: str) -> np.ndarray:
        """Return the world-space position of an arm's gripper site."""
        if arm not in self.arm_names:
            raise ValueError(
                f"Unknown arm '{arm}'. Configured arms: {self.arm_names}"
            )
        return self.get_site_position(f"{arm}_gripper_site")

    # ----------------------------------------------------------------------
    # Global Joint State & Targeting API (Backward Compatibility)
    # ----------------------------------------------------------------------

    def get_joint_positions(self, in_degrees: bool = True) -> Dict[str, Union[float, List[float]]]:
        """Return current positions for all joints in the model.

        Hinge joints are returned in degrees if in_degrees is True.
        Free joints (e.g. plate_joint) return [x, y, z, qw, qx, qy, qz].
        """
        positions: Dict[str, Union[float, List[float]]] = {}
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[jid]
            j_type = self.model.jnt_type[jid]
            if j_type == mujoco.mjtJoint.mjJNT_FREE:
                positions[name] = [float(v) for v in self.data.qpos[qpos_adr : qpos_adr + 7]]
            elif j_type == mujoco.mjtJoint.mjJNT_HINGE:
                val = float(self.data.qpos[qpos_adr])
                positions[name] = float(np.degrees(val) if in_degrees else val)
            else:
                positions[name] = float(self.data.qpos[qpos_adr])
        return positions

    def get_joint_velocities(self, in_degrees: bool = True) -> Dict[str, Union[float, List[float]]]:
        """Return current velocities for all joints."""
        velocities: Dict[str, Union[float, List[float]]] = {}
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof_adr = self.model.jnt_dofadr[jid]
            j_type = self.model.jnt_type[jid]
            if j_type == mujoco.mjtJoint.mjJNT_FREE:
                velocities[name] = [float(v) for v in self.data.qvel[dof_adr : dof_adr + 6]]
            elif j_type == mujoco.mjtJoint.mjJNT_HINGE:
                val = float(self.data.qvel[dof_adr])
                velocities[name] = float(np.degrees(val) if in_degrees else val)
            else:
                velocities[name] = float(self.data.qvel[dof_adr])
        return velocities

    def set_joint_targets(self, targets: Dict[str, float], in_degrees: bool = True) -> None:
        """Apply position control targets across all actuators.

        Args:
            targets: Dictionary mapping actuator or joint names to target values.
            in_degrees: If True (default), targets are in degrees and converted to radians.
        """
        for name, value in targets.items():
            matched_aid: Optional[int] = None
            if name in self.actuator_names:
                matched_aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            else:
                act_name = f"actuator_{name}" if not name.startswith("actuator_") else name
                if act_name in self.actuator_names:
                    matched_aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)

            if matched_aid is not None:
                # Convert hinge position targets from degrees to radians
                ctrl_val = float(np.radians(value) if in_degrees else value)
                self.data.ctrl[matched_aid] = ctrl_val

    # ----------------------------------------------------------------------
    # Camera Rendering API
    # ----------------------------------------------------------------------

    def get_camera_names(self) -> List[str]:
        """Return list of available camera names in the simulation scene."""
        return list(self.camera_names)

    def get_camera_image(
        self,
        camera_name: str = "overview_cam",
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> np.ndarray:
        """Render and return an RGB image from the specified scene camera.

        Args:
            camera_name: Name of camera in MJCF (e.g. 'overview_cam', 'front_cam').
            width: Image width in pixels. If None, loaded from sim_config.yaml.
            height: Image height in pixels. If None, loaded from sim_config.yaml.

        Returns:
            np.ndarray: uint8 array of shape (height, width, 3).
        """
        if camera_name not in self.camera_names:
            raise ValueError(
                f"Camera '{camera_name}' not found. Available cameras: {self.camera_names}"
            )

        cam_cfg = self.config.get("cameras", {}).get(camera_name, {})
        if width is None:
            width = int(cam_cfg.get("width", 640))
        if height is None:
            height = int(cam_cfg.get("height", 480))

        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)

        renderer = self._renderers[key]
        renderer.update_scene(self.data, camera=camera_name)
        return renderer.render()

    def close(self) -> None:
        """Clean up rendering contexts and resources."""
        self._renderers.clear()
