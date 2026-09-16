"""Ground-truth scene state extraction for TwinGuard simulation.

Directly introspects MuJoCo physics state to extract positions, orientations,
and configurations for scene objects (plate, mug, drawer) and both robotic arms
(left_arm, right_arm), and serializes them into a compact text format suitable
for LLM prompting.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import numpy as np
import mujoco

from simulation.simulator import TwinGuardSim


# Supported scene objects for milestone scope
TRACKED_OBJECTS = ("plate", "mug", "drawer")


@dataclass
class ObjectState:
    """Ground truth state of a manipulated or static scene object."""

    name: str
    position: List[float]  # [x, y, z] in world coordinates
    orientation: List[float]  # [w, x, y, z] quaternion
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArmState:
    """Ground truth state of a robotic manipulator."""

    name: str
    joint_positions: Dict[str, float]  # canonical joint names -> degrees
    gripper_position: float  # degrees (0 = closed, ~40 = open)
    gripper_state: str  # "open" or "closed"
    ee_position: List[float]  # [x, y, z] gripper site coordinates


@dataclass
class SceneState:
    """Complete ground-truth scene description."""

    objects: Dict[str, ObjectState]
    arms: Dict[str, ArmState]
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "objects": {
                name: {
                    "name": obj.name,
                    "position": obj.position,
                    "orientation": obj.orientation,
                    "extra": obj.extra,
                }
                for name, obj in self.objects.items()
            },
            "arms": {
                name: {
                    "name": arm.name,
                    "joint_positions": arm.joint_positions,
                    "gripper_position": arm.gripper_position,
                    "gripper_state": arm.gripper_state,
                    "ee_position": arm.ee_position,
                }
                for name, arm in self.arms.items()
            },
        }


def get_ground_truth_scene_state(sim: TwinGuardSim) -> SceneState:
    """Read ground-truth object poses and robot arm states directly from MuJoCo.

    Args:
        sim: Active TwinGuardSim simulation instance.

    Returns:
        SceneState containing tracked objects and arm states.
    """
    objects: Dict[str, ObjectState] = {}
    for obj_name in TRACKED_OBJECTS:
        bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, obj_name)
        if bid == -1:
            continue

        pos = [round(float(v), 3) for v in sim.data.xpos[bid]]
        quat = [round(float(v), 3) for v in sim.data.xquat[bid]]
        extra: Dict[str, Any] = {}

        # Special attributes for drawer
        if obj_name == "drawer":
            drawer_jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_joint")
            if drawer_jid != -1:
                qpos_adr = sim.model.jnt_qposadr[drawer_jid]
                slide_val = float(sim.data.qpos[qpos_adr])
                extra["slide_position"] = round(slide_val, 3)
                extra["is_open"] = bool(slide_val > 0.02)

            handle_sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, "drawer_handle_site")
            if handle_sid != -1:
                extra["handle_position"] = [round(float(v), 3) for v in sim.data.site_xpos[handle_sid]]

        objects[obj_name] = ObjectState(
            name=obj_name,
            position=pos,
            orientation=quat,
            extra=extra,
        )

    arms: Dict[str, ArmState] = {}
    for arm_name in ("left_arm", "right_arm"):
        if arm_name not in sim.arm_names:
            continue

        full_joints = sim.get_arm_joint_positions(arm_name, in_degrees=True)
        # Filter canonical 5 joints for compact summary
        canonical_5 = (
            "joint_base_yaw",
            "joint_shoulder_pitch",
            "joint_elbow_pitch",
            "joint_wrist_pitch",
            "joint_wrist_roll",
        )
        canonical_joints = {
            cj.removeprefix("joint_"): round(float(full_joints.get(cj, 0.0)), 1)
            for cj in canonical_5
            if cj in full_joints
        }

        grip_pos = float(sim.get_arm_gripper_position(arm_name, in_degrees=True))
        grip_state = "open" if grip_pos > 20.0 else "closed"

        site_name = f"{arm_name}_gripper_site"
        sid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        ee_pos = [round(float(v), 3) for v in sim.data.site_xpos[sid]] if sid != -1 else [0.0, 0.0, 0.0]

        arms[arm_name] = ArmState(
            name=arm_name,
            joint_positions=canonical_joints,
            gripper_position=round(grip_pos, 1),
            gripper_state=grip_state,
            ee_position=ee_pos,
        )

    return SceneState(
        objects=objects,
        arms=arms,
        timestamp=round(sim.get_time(), 3),
    )


def format_scene_state(sim_or_state: Union[TwinGuardSim, SceneState, Dict[str, Any]]) -> str:
    """Format ground-truth scene state into a compact text prompt for an LLM.

    Args:
        sim_or_state: TwinGuardSim instance, SceneState dataclass, or dictionary.

    Returns:
        Compact multi-line string representation.
    """
    if isinstance(sim_or_state, TwinGuardSim):
        state = get_ground_truth_scene_state(sim_or_state)
    elif isinstance(sim_or_state, SceneState):
        state = sim_or_state
    elif isinstance(sim_or_state, dict):
        lines = ["CURRENT SCENE STATE:"]
        lines.append("Objects:")
        for name, obj in sim_or_state.get("objects", {}).items():
            pos = obj.get("position", [])
            quat = obj.get("orientation", [])
            extra_info = ""
            extra = obj.get("extra", {})
            if "slide_position" in extra:
                status = "open" if extra.get("is_open") else "closed"
                extra_info = f", slide={extra['slide_position']}m ({status})"
            lines.append(f"  - {name}: pos={pos}, quat={quat}{extra_info}")

        lines.append("Robots / Arms:")
        for name, arm in sim_or_state.get("arms", {}).items():
            alias = "arm A" if name == "left_arm" else "arm B"
            ee = arm.get("ee_position", [])
            grip = arm.get("gripper_position", 0.0)
            status = arm.get("gripper_state", "closed")
            joints = arm.get("joint_positions", {})
            joints_str = ", ".join(f"{k}: {v}" for k, v in joints.items())
            lines.append(f"  - {name} ({alias}): ee_pos={ee}, gripper={grip} deg ({status}), joints={{{joints_str}}}")

        return "\n".join(lines)
    else:
        raise TypeError(f"Unsupported scene state type: {type(sim_or_state)}")

    lines = ["CURRENT SCENE STATE:"]
    lines.append("Objects:")
    for name, obj in state.objects.items():
        extra_info = ""
        if "slide_position" in obj.extra:
            status = "open" if obj.extra.get("is_open") else "closed"
            extra_info = f", slide={obj.extra['slide_position']}m ({status})"
        lines.append(f"  - {name}: pos={obj.position}, quat={obj.orientation}{extra_info}")

    lines.append("Robots / Arms:")
    for name, arm in state.arms.items():
        alias = "arm A" if name == "left_arm" else "arm B"
        joints_str = ", ".join(f"{k}: {v}" for k, v in arm.joint_positions.items())
        lines.append(
            f"  - {name} ({alias}): ee_pos={arm.ee_position}, gripper={arm.gripper_position} deg ({arm.gripper_state}), joints={{{joints_str}}}"
        )

    return "\n".join(lines)


def get_scene_state_prompt(sim: TwinGuardSim) -> str:
    """Convenience helper to extract and format scene state for an LLM prompt."""
    return format_scene_state(sim)
