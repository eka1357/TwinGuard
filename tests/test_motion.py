import pytest

from robotics.motion import move_arm_to_targets


class FakeSim:
    def __init__(self, stable=True):
        self.stable = stable
        self.target_calls = []
        self.step_calls = 0
        self.stability_checks = 0

    def set_arm_joint_targets(self, arm, targets):
        self.target_calls.append((arm, targets))

    def step(self, n_steps):
        assert n_steps == 1
        self.step_calls += 1

    def is_stable(self):
        self.stability_checks += 1
        return self.stable


def test_targets_are_sent_to_correct_arm():
    sim = FakeSim()
    targets = {"joint_shoulder_pitch": 15.0}

    move_arm_to_targets(sim, "left_arm", targets, steps=3)

    assert sim.target_calls == [("left_arm", targets)]


def test_simulator_is_stepped_requested_number_of_times():
    sim = FakeSim()

    move_arm_to_targets(sim, "right_arm", {}, steps=4)

    assert sim.step_calls == 4
    assert sim.stability_checks == 4


@pytest.mark.parametrize("steps", [0, -1])
def test_non_positive_steps_raise_value_error(steps):
    sim = FakeSim()

    with pytest.raises(ValueError, match="greater than zero"):
        move_arm_to_targets(sim, "left_arm", {}, steps=steps)

    assert sim.target_calls == []
    assert sim.step_calls == 0


def test_unstable_simulation_raises_and_stops_stepping():
    sim = FakeSim(stable=False)

    with pytest.raises(RuntimeError, match="left_arm"):
        move_arm_to_targets(sim, "left_arm", {}, steps=5)

    assert sim.step_calls == 1
    assert sim.stability_checks == 1
