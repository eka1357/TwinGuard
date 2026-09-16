import numpy as np
import pytest

from robotics.grasp_checks import (
    is_gripper_near_plate,
    require_gripper_near_plate,
)


class FakeSim:
    def __init__(self, gripper_position):
        self.gripper_position = np.asarray(gripper_position, dtype=float)
        self.actions = []

    def get_joint_positions(self):
        return {
            "plate_joint": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        }

    def get_gripper_position(self, arm):
        return self.gripper_position

    def open_gripper(self, arm):
        self.actions.append("open_gripper")

    def close_gripper(self, arm):
        self.actions.append("close_gripper")

    def step(self, n_steps):
        self.actions.append("step")


def test_gripper_within_threshold_returns_true():
    sim = FakeSim([0.0, 0.0, 0.05])

    assert is_gripper_near_plate(sim, "left_arm", max_distance=0.08)


def test_gripper_outside_threshold_returns_false():
    sim = FakeSim([0.0, 0.0, 0.10])

    assert not is_gripper_near_plate(sim, "left_arm", max_distance=0.08)


def test_require_gripper_near_plate_returns_distance_when_close():
    sim = FakeSim([0.0, 0.0, 0.05])

    distance = require_gripper_near_plate(
        sim,
        "right_arm",
        max_distance=0.08,
    )

    assert distance == pytest.approx(0.05)


def test_require_gripper_near_plate_raises_when_too_far():
    sim = FakeSim([0.0, 0.0, 0.10])

    with pytest.raises(ValueError, match="too far"):
        require_gripper_near_plate(sim, "left_arm", max_distance=0.08)


@pytest.mark.parametrize("function", [
    is_gripper_near_plate,
    require_gripper_near_plate,
])
def test_non_positive_max_distance_raises(function):
    sim = FakeSim([0.0, 0.0, 0.05])

    with pytest.raises(ValueError, match="greater than zero"):
        function(sim, "left_arm", max_distance=0.0)


def test_proximity_helpers_do_not_control_simulator():
    sim = FakeSim([0.0, 0.0, 0.05])

    is_gripper_near_plate(sim, "left_arm")
    require_gripper_near_plate(sim, "left_arm")

    assert sim.actions == []
