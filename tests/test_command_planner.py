import pytest

from planning.command_planner import (
    prepare_command,
    prepare_commands,
    validate_command,
    SUPPORTED_ACTIONS,
    SUPPORTED_OBJECTS,
)

def test_prepare_command_basic_transformation():
    command = {
        "arm": "A",
        "action": "pick",
        "obj": "plate",
        "target": None,
    }
    expected = {
        "arm": "A",
        "sim_arm": "left_arm",
        "action": "pick",
        "obj": "plate",
        "target": None,
    }
    result = prepare_command(command)
    assert result == expected
    assert list(result.keys()) == list(expected.keys())


def test_prepare_command_arm_b():
    command = {"arm": "B", "action": "place", "obj": "mug", "target": "table"}
    result = prepare_command(command)
    assert result["sim_arm"] == "right_arm"


def test_prepare_command_does_not_mutate_input():
    command = {"arm": "A", "action": "pick", "obj": "plate", "target": None}
    original_copy = dict(command)
    prepare_command(command)
    assert command == original_copy
    assert "sim_arm" not in command


def test_prepare_command_missing_arm_key_raises():
    command = {"action": "pick", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        prepare_command(command)


def test_prepare_command_invalid_arm_raises():
    command = {"arm": "C", "action": "pick", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        prepare_command(command)


def test_prepare_command_none_arm_raises():
    command = {"arm": None, "action": "pick", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        prepare_command(command)


def test_prepare_commands_preserves_order_and_does_not_mutate():
    commands = [
        {"arm": "A", "action": "pick", "obj": "plate", "target": None},
        {"arm": "B", "action": "place", "obj": "mug", "target": "table"},
    ]
    originals = [dict(c) for c in commands]

    results = prepare_commands(commands)

    assert [r["sim_arm"] for r in results] == ["left_arm", "right_arm"]
    assert [r["arm"] for r in results] == ["A", "B"]
    assert commands == originals  # inputs untouched


def test_prepare_commands_empty_list():
    assert prepare_commands([]) == []



def test_validate_command_valid_passes():
    command = {"arm": "A", "action": "pick", "obj": "plate", "target": None}
    assert validate_command(command) is None


@pytest.mark.parametrize("action", sorted(SUPPORTED_ACTIONS))
def test_validate_command_all_supported_actions(action):
    command = {"arm": "A", "action": action, "obj": "plate", "target": None}
    assert validate_command(command) is None


@pytest.mark.parametrize("obj", sorted(SUPPORTED_OBJECTS))
def test_validate_command_all_supported_objects(obj):
    command = {"arm": "B", "action": "pick", "obj": obj, "target": None}
    assert validate_command(command) is None


def test_validate_command_missing_arm_raises():
    command = {"action": "pick", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        validate_command(command)


def test_validate_command_invalid_arm_raises():
    command = {"arm": "C", "action": "pick", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        validate_command(command)


def test_validate_command_missing_action_raises():
    command = {"arm": "A", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        validate_command(command)


def test_validate_command_unsupported_action_raises():
    command = {"arm": "A", "action": "dance", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        validate_command(command)


def test_validate_command_missing_obj_raises():
    command = {"arm": "A", "action": "pick", "target": None}
    with pytest.raises(ValueError):
        validate_command(command)


def test_validate_command_unsupported_obj_raises():
    command = {"arm": "A", "action": "pick", "obj": "spoon", "target": None}
    with pytest.raises(ValueError):
        validate_command(command)


def test_prepare_command_still_works_after_validation_added():
    command = {"arm": "A", "action": "pick", "obj": "plate", "target": None}
    result = prepare_command(command)
    assert result["sim_arm"] == "left_arm"

def test_prepare_command_unsupported_action_raises():
    command = {"arm": "A", "action": "dance", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        prepare_command(command)


def test_prepare_command_unsupported_obj_raises():
    command = {"arm": "A", "action": "pick", "obj": "spoon", "target": None}
    with pytest.raises(ValueError):
        prepare_command(command)


def test_prepare_command_unsupported_action_and_obj_raises():
    command = {"arm": "A", "action": "dance", "obj": "spoon", "target": None}
    with pytest.raises(ValueError):
        prepare_command(command)


def test_prepare_command_missing_action_raises():
    command = {"arm": "A", "obj": "plate", "target": None}
    with pytest.raises(ValueError):
        prepare_command(command)


def test_prepare_command_missing_obj_raises():
    command = {"arm": "A", "action": "pick", "target": None}
    with pytest.raises(ValueError):
        prepare_command(command)


def test_prepare_command_does_not_mutate_input_on_invalid_action():
    command = {"arm": "A", "action": "dance", "obj": "plate", "target": None}
    original_copy = dict(command)
    with pytest.raises(ValueError):
        prepare_command(command)
    assert command == original_copy


def test_prepare_commands_raises_on_first_invalid_item():
    commands = [
        {"arm": "A", "action": "pick", "obj": "plate", "target": None},
        {"arm": "B", "action": "dance", "obj": "mug", "target": None},
    ]
    with pytest.raises(ValueError):
        prepare_commands(commands)