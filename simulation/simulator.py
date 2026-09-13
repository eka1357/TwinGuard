"""Core MuJoCo simulation environment for TwinGuard."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import mujoco


class TwinGuardSim:
    """TwinGuard MuJoCo simulation wrapper for the SO-101 robotic arm."""

    DEFAULT_SCENE_PATH = Path(__file__).resolve().parent / "models" / "scene.xml"

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the MuJoCo simulation.

        Args:
            model_path: Optional path to an MJCF XML file. Defaults to scene.xml.
        """
        self.model_path = Path(model_path) if model_path else self.DEFAULT_SCENE_PATH
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model file not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        self.joint_names: List[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.actuator_names: List[str] = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

    def reset(self) -> None:
        """Reset simulation state to the keyframe / initial zero configuration."""
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
        """Return the current simulation time in seconds."""
        return float(self.data.time)

    def get_timestep(self) -> float:
        """Return the physics timestep (dt)."""
        return float(self.model.opt.timestep)

    def get_joint_positions(self) -> Dict[str, float]:
        """Return current positions for all joints (in degrees for hinge joints)."""
        positions = {}
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_adr = self.model.jnt_qposadr[jid]
            positions[name] = float(self.data.qpos[qpos_adr])
        return positions

    def get_joint_velocities(self) -> Dict[str, float]:
        """Return current velocities for all joints."""
        velocities = {}
        for name in self.joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dof_adr = self.model.jnt_dofadr[jid]
            velocities[name] = float(self.data.qvel[dof_adr])
        return velocities

    def set_joint_targets(self, targets: Dict[str, float]) -> None:
        """Apply position control targets to actuators.

        Args:
            targets: Dictionary mapping actuator or joint names to target values.
        """
        for name, value in targets.items():
            if name in self.actuator_names:
                aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                self.data.ctrl[aid] = float(value)
            else:
                # Try matching by actuator_<joint_name> or joint_<name>
                act_name = f"actuator_{name}" if not name.startswith("actuator_") else name
                if act_name in self.actuator_names:
                    aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
                    self.data.ctrl[aid] = float(value)

    def is_stable(self) -> bool:
        """Check if the simulation state is physically stable (no NaNs or infinities)."""
        qpos_valid = not np.any(np.isnan(self.data.qpos)) and not np.any(np.isinf(self.data.qpos))
        qvel_valid = not np.any(np.isnan(self.data.qvel)) and not np.any(np.isinf(self.data.qvel))
        return bool(qpos_valid and qvel_valid)
