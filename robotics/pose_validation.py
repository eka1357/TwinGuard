from typing import Mapping

JOINT_LIMITS = {
    "joint_base_yaw": (-150.0, 150.0),
    "joint_shoulder_pitch": (-110.0, 110.0),
    "joint_elbow_pitch": (-120.0, 120.0),
    "joint_wrist_pitch": (-110.0, 110.0),
    "joint_wrist_roll": (-180.0, 180.0),
    "joint_gripper": (0.0, 40.0),
}


def validate_pose(pose: Mapping[str, float]) -> None:
    """Validate canonical joint names and degree limits."""
    unknown_joints = set(pose) - set(JOINT_LIMITS)
    if unknown_joints:
        raise ValueError(f"Unknown joints: {sorted(unknown_joints)}")

    for joint, value in pose.items():
        lower, upper = JOINT_LIMITS[joint]
        if not lower <= value <= upper:
            raise ValueError(
                f"Joint {joint!r} target {value} is outside "
                f"[{lower}, {upper}] degrees."
            )
