import numpy as np
import pytest

from robotics.reachability import get_plate_position, gripper_to_plate_distance


class FakeSim:
    def __init__(self):
        self.requested_arms = []

    def get_joint_positions(self):
        return {
            "plate_joint": [0.25, 0.0, 0.435, 1.0, 0.0, 0.0, 0.0]
        }

    def get_gripper_position(self, arm):
        self.requested_arms.append(arm)
        if arm == "left_arm":
            return np.array([0.25, 0.0, 0.535])
        return np.array([0.25, 0.0, 0.635])


def test_get_plate_position_returns_xyz_array():
    sim = FakeSim()

    position = get_plate_position(sim)

    assert isinstance(position, np.ndarray)
    assert position.shape == (3,)
    np.testing.assert_array_equal(position, [0.25, 0.0, 0.435])


def test_gripper_to_plate_distance_returns_correct_float():
    sim = FakeSim()

    distance = gripper_to_plate_distance(sim, "left_arm")

    assert isinstance(distance, float)
    assert distance == pytest.approx(0.1)


def test_different_arm_name_is_forwarded():
    sim = FakeSim()

    distance = gripper_to_plate_distance(sim, "right_arm")

    assert sim.requested_arms == ["right_arm"]
    assert distance == pytest.approx(0.2)
