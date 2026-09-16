"""Smoke test for executing the validated home pose on both arms."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.motion import move_arm_to_targets
from robotics.pose_validation import validate_pose
from robotics.poses import HOME_POSE
from simulation.simulator import TwinGuardSim


def main() -> None:
    validate_pose(HOME_POSE)

    sim = TwinGuardSim()

    try:
        sim.reset()

        move_arm_to_targets(sim, "left_arm", HOME_POSE, steps=100)
        move_arm_to_targets(sim, "right_arm", HOME_POSE, steps=100)

        assert sim.is_stable()
        print("[SUCCESS] HOME_POSE executed for both arms.")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
