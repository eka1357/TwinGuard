import pytest

from robotics.pose_validation import JOINT_LIMITS, validate_pose
from robotics.poses import HOME_POSE


def test_home_pose_passes_validation():
    validate_pose(HOME_POSE)


def test_valid_partial_pose_passes_validation():
    validate_pose({"joint_shoulder_pitch": 25.0, "joint_gripper": 10.0})


def test_unknown_joint_raises_value_error():
    with pytest.raises(ValueError, match="Unknown joints"):
        validate_pose({"joint_unknown": 0.0})


@pytest.mark.parametrize(
    ("joint", "value"),
    [
        (joint, lower - 0.1)
        for joint, (lower, _upper) in JOINT_LIMITS.items()
    ],
)
def test_value_below_lower_limit_raises_value_error(joint, value):
    with pytest.raises(ValueError, match="outside"):
        validate_pose({joint: value})


@pytest.mark.parametrize(
    ("joint", "value"),
    [
        (joint, upper + 0.1)
        for joint, (_lower, upper) in JOINT_LIMITS.items()
    ],
)
def test_value_above_upper_limit_raises_value_error(joint, value):
    with pytest.raises(ValueError, match="outside"):
        validate_pose({joint: value})


@pytest.mark.parametrize(
    ("joint", "value"),
    [
        (joint, boundary)
        for joint, (lower, upper) in JOINT_LIMITS.items()
        for boundary in (lower, upper)
    ],
)
def test_exact_limits_are_accepted(joint, value):
    validate_pose({joint: value})


def test_input_pose_is_not_modified():
    pose = {
        "joint_base_yaw": 10.0,
        "joint_gripper": 20.0,
    }
    original_pose = pose.copy()

    validate_pose(pose)

    assert pose == original_pose
