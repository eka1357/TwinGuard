"""Smoke test for joint-space motion against the real simulator."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.motion import move_arm_to_targets
from simulation.simulator import TwinGuardSim


def main() -> None:
    sim = TwinGuardSim()

    try:
        sim.reset()

        print("Initial left arm position:")
        print(sim.get_arm_joint_positions("left_arm"))

        move_arm_to_targets(
            sim,
            "left_arm",
            {
                "joint_base_yaw": 10.0,
                "joint_shoulder_pitch": -5.0,
            },
            steps=100,
        )

        print("Final left arm position:")
        print(sim.get_arm_joint_positions("left_arm"))

        print("Left gripper position:")
        print(sim.get_gripper_position("left_arm"))

        assert sim.is_stable()
        print("[SUCCESS] Real joint-space motion test passed.")

    finally:
        sim.close()


if __name__ == "__main__":
    main()