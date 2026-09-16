import pytest

from robotics.command_executor import execute_command


class FakeSim:
    """Records gripper calls without touching MuJoCo."""

    def __init__(self):
        self.calls = []

    def open_gripper(self, arm):
        self.calls.append(("open_gripper", arm))

    def close_gripper(self, arm):
        self.calls.append(("close_gripper", arm))


def make_command(action, obj="plate", sim_arm="left_arm", arm="A"):
    return {
        "arm": arm,
        "sim_arm": sim_arm,
        "action": action,
        "obj": obj,
        "target": None,
    }


def test_open_calls_open_gripper():
    sim = FakeSim()
    command = make_command("open")
    execute_command(sim, command)
    assert sim.calls == [("open_gripper", "left_arm")]


def test_pick_closes_gripper():
    sim = FakeSim()
    command = make_command("pick")
    execute_command(sim, command)
    assert sim.calls == [("close_gripper", "left_arm")]


def test_place_opens_gripper():
    sim = FakeSim()
    command = make_command("place")
    execute_command(sim, command)
    assert sim.calls == [("open_gripper", "left_arm")]


def test_pick_uses_correct_arm_for_right():
    sim = FakeSim()
    command = make_command("pick", sim_arm="right_arm", arm="B")
    execute_command(sim, command)
    assert sim.calls == [("close_gripper", "right_arm")]


def test_pour_raises_not_implemented():
    sim = FakeSim()
    command = make_command("pour")
    with pytest.raises(NotImplementedError):
        execute_command(sim, command)
    assert sim.calls == []


def test_handoff_raises_not_implemented():
    sim = FakeSim()
    command = make_command("handoff")
    with pytest.raises(NotImplementedError):
        execute_command(sim, command)
    assert sim.calls == []


def test_missing_sim_arm_raises_value_error():
    sim = FakeSim()
    command = {"arm": "A", "action": "pick", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        execute_command(sim, command)
    assert sim.calls == []


def test_none_sim_arm_raises_value_error():
    sim = FakeSim()
    command = make_command("pick", sim_arm=None)
    with pytest.raises(ValueError):
        execute_command(sim, command)
    assert sim.calls == []


def test_object_not_in_scene_raises_value_error():
    sim = FakeSim()
    command = make_command("pick", obj="mug")  # mug is a valid parser object,
    # but scene_capabilities currently only models plate/table
    with pytest.raises(ValueError):
        execute_command(sim, command)
    assert sim.calls == []


def test_object_in_scene_table_with_open_action():
    sim = FakeSim()
    command = make_command("open", obj="table")
    execute_command(sim, command)
    assert sim.calls == [("open_gripper", "left_arm")]

def test_missing_object_raises_value_error():
    sim = FakeSim()
    command = {
        "sim_arm": "left_arm",
        "action": "open",
    }

    with pytest.raises(ValueError, match="obj"):
        execute_command(sim, command)

    assert sim.calls == []


def test_missing_action_raises_value_error():
    sim = FakeSim()
    command = {
        "sim_arm": "left_arm",
        "obj": "plate",
    }

    with pytest.raises(ValueError, match="action"):
        execute_command(sim, command)

    assert sim.calls == []


def test_unsupported_action_raises_value_error():
    sim = FakeSim()
    command = {
        "sim_arm": "left_arm",
        "action": "dance",
        "obj": "plate",
    }

    with pytest.raises(ValueError, match="Unsupported action"):
        execute_command(sim, command)

    assert sim.calls == []