"""Bridges parser command dictionaries to simulator-ready commands.

Takes parser output (using arm labels "A"/"B") and adds the
corresponding simulator arm name ("left_arm"/"right_arm") without
mutating the original command or losing field order semantics.
"""

from planning.arm_mapping import parser_arm_to_sim_arm

SUPPORTED_ACTIONS = {"pick", "place", "pour", "open", "handoff"}
SUPPORTED_OBJECTS = {"plate", "mug", "drawer"}

def validate_command(command: dict) -> None:
    """Validate a parser command dictionary.

    Checks that ``arm``, ``action``, and ``obj`` are present and
    valid/supported. Does not check ``target`` and does not verify
    that objects physically exist in the simulation scene.

    Raises:
        ValueError: if any required field is missing or invalid.
    """
    if "arm" not in command:
        raise ValueError("Command is missing required 'arm' field.")
    parser_arm_to_sim_arm(command["arm"])  # raises ValueError if invalid

    if "action" not in command:
        raise ValueError("Command is missing required 'action' field.")
    action = command["action"]
    if action not in SUPPORTED_ACTIONS:
        raise ValueError(
            f"Unsupported action: {action!r}. "
            f"Expected one of {sorted(SUPPORTED_ACTIONS)}."
        )

    if "obj" not in command:
        raise ValueError("Command is missing required 'obj' field.")
    obj = command["obj"]
    if obj not in SUPPORTED_OBJECTS:
        raise ValueError(
            f"Unsupported object: {obj!r}. "
            f"Expected one of {sorted(SUPPORTED_OBJECTS)}."
        )

def prepare_command(command: dict) -> dict:
    """Return a new command dict with a sim_arm field added.

    The command is validated first via validate_command(). The
    original ``command`` dict is never modified. The returned dict
    preserves the original ``arm`` field and inserts ``sim_arm``
    immediately after it, followed by the remaining original keys
    in their original order.

    Raises:
        ValueError: if the command fails validation (missing/invalid
            arm, action, or obj).
    """
    validate_command(command)
    sim_arm = parser_arm_to_sim_arm(command["arm"])

    result = {}
    for key, value in command.items():
        result[key] = value
        if key == "arm":
            result["sim_arm"] = sim_arm

    return result


def prepare_commands(commands: list) -> list:
    """Apply prepare_command to a list of commands, preserving order.

    Returns a new list of new dicts. The input list and its
    dictionaries are never modified.
    """
    return [prepare_command(cmd) for cmd in commands]