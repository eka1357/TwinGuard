"""Manual gripper-pose calibration probe for the TwinGuard simulator."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.motion import move_arm_to_targets
from robotics.pose_validation import validate_pose
from robotics.poses import HOME_POSE
from robotics.reachability import gripper_to_plate_distance
from simulation.simulator import TwinGuardSim


CANDIDATE_POSES = {
    "left_low_reach": {
        "joint_base_yaw": 20.0,
        "joint_shoulder_pitch": -30.0,
        "joint_elbow_pitch": 60.0,
        "joint_wrist_pitch": -30.0,
        "joint_wrist_roll": 0.0,
        "joint_gripper": 40.0,
    },
    "right_low_reach": {
        "joint_base_yaw": -20.0,
        "joint_shoulder_pitch": -30.0,
        "joint_elbow_pitch": 60.0,
        "joint_wrist_pitch": -30.0,
        "joint_wrist_roll": 0.0,
        "joint_gripper": 40.0,
    },
}

REACHABILITY_THRESHOLD = 0.05


def main() -> None:
    validate_pose(HOME_POSE)
    for pose in CANDIDATE_POSES.values():
        validate_pose(pose)

    sim = TwinGuardSim()

    try:
        sim.reset()

        print("Plate position:")
        print(sim.get_joint_positions()["plate_joint"])

        for arm in ("left_arm", "right_arm"):
            move_arm_to_targets(sim, arm, HOME_POSE, steps=100)

        print("Left gripper position:")
        print(sim.get_gripper_position("left_arm"))
        print("Right gripper position:")
        print(sim.get_gripper_position("right_arm"))

        for candidate_name, pose in CANDIDATE_POSES.items():
            if candidate_name.startswith("left_"):
                arm = "left_arm"
            elif candidate_name.startswith("right_"):
                arm = "right_arm"
            else:
                raise ValueError(
                    f"Candidate pose name must start with 'left_' or 'right_': "
                    f"{candidate_name}"
                )

            move_arm_to_targets(sim, arm, pose, steps=200)
            gripper_position = sim.get_gripper_position(arm)
            distance = gripper_to_plate_distance(sim, arm)
            status = (
                "REACHABLE"
                if distance <= REACHABILITY_THRESHOLD
                else "TOO_FAR"
            )
            print(f"Candidate: {candidate_name}")
            print(f"Arm: {arm}")
            print(f"Gripper: {gripper_position}")
            print(f"Plate: {sim.get_joint_positions()['plate_joint'][:3]}")
            print(f"Distance: {distance:.4f} m")
            print(f"Status: {status}")
            assert sim.is_stable()

        move_arm_to_targets(sim, "left_arm", HOME_POSE, steps=100)
        move_arm_to_targets(sim, "right_arm", HOME_POSE, steps=100)

        print("[SUCCESS] Gripper pose calibration report completed.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
