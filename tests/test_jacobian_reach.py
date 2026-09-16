from types import SimpleNamespace

import numpy as np
import pytest

import robotics.jacobian_reach as jacobian_reach
from robotics.jacobian_reach import move_gripper_toward


class MinimalSim:
    arm_names = ["left_arm"]

    def __init__(self):
        self.model = SimpleNamespace(
            nv=6,
            jnt_dofadr=np.arange(6),
        )
        self.data = SimpleNamespace(
            site_xpos=np.zeros((1, 3)),
        )
        self.targets = []
        self.steps = 0

    def get_arm_joint_positions(self, arm):
        return {
            "joint_base_yaw": 0.0,
            "joint_shoulder_pitch": 0.0,
            "joint_elbow_pitch": 0.0,
            "joint_wrist_pitch": 0.0,
            "joint_wrist_roll": 0.0,
            "joint_gripper": 40.0,
        }

    def set_arm_joint_targets(self, arm, targets):
        self.targets.append((arm, targets))

    def step(self, n_steps):
        self.steps += n_steps

    def is_stable(self):
        return True


@pytest.fixture
def fake_mujoco(monkeypatch):
    joint_ids = {
        "left_arm_joint_base_yaw": 0,
        "left_arm_joint_shoulder_pitch": 1,
        "left_arm_joint_elbow_pitch": 2,
        "left_arm_joint_wrist_pitch": 3,
        "left_arm_joint_wrist_roll": 4,
    }

    def name_to_id(_model, object_type, name):
        if object_type == jacobian_reach.mujoco.mjtObj.mjOBJ_SITE:
            return 0 if name == "left_arm_gripper_site" else -1
        return joint_ids.get(name, -1)

    def jac_site(_model, _data, jacp, jacr, _site_id):
        jacp.fill(0.0)
        jacr.fill(0.0)
        jacp[0, 0] = 1.0

    monkeypatch.setattr(jacobian_reach.mujoco, "mj_name2id", name_to_id)
    monkeypatch.setattr(jacobian_reach.mujoco, "mj_jacSite", jac_site)


def test_invalid_arm():
    sim = MinimalSim()

    with pytest.raises(ValueError, match="Unknown arm"):
        move_gripper_toward(sim, "right_arm", np.zeros(3))


def test_invalid_target_shape():
    sim = MinimalSim()

    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        move_gripper_toward(sim, "left_arm", np.zeros(2))


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("max_iterations", 0, "max_iterations"),
        ("position_tolerance", 0, "position_tolerance"),
        ("step_scale", 0, "step_scale"),
        ("steps_per_update", 0, "steps_per_update"),
        ("damping", 0, "damping"),
    ],
)
def test_invalid_reach_parameters(argument, value, message):
    sim = MinimalSim()

    with pytest.raises(ValueError, match=message):
        move_gripper_toward(
            sim,
            "left_arm",
            np.zeros(3),
            **{argument: value},
        )


def test_gripper_joint_is_never_included_in_updates(fake_mujoco):
    sim = MinimalSim()

    move_gripper_toward(
        sim,
        "left_arm",
        np.array([1.0, 0.0, 0.0]),
        max_iterations=1,
    )

    assert sim.targets
    _, targets = sim.targets[0]
    assert set(targets) == {
        "joint_base_yaw",
        "joint_shoulder_pitch",
        "joint_elbow_pitch",
        "joint_wrist_pitch",
        "joint_wrist_roll",
    }
    assert "joint_gripper" not in targets


def test_steps_per_update_controls_physics_steps(fake_mujoco):
    sim = MinimalSim()

    move_gripper_toward(
        sim,
        "left_arm",
        np.array([1.0, 0.0, 0.0]),
        max_iterations=1,
        steps_per_update=7,
    )

    assert sim.steps == 7
