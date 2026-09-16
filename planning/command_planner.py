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


from typing import Dict, List, Optional
from planning.planner import PlanStep

DEFAULT_OBJECT_COORDS: Dict[str, List[float]] = {
    "plate": [0.25, 0.0, 0.435],
    "mug": [0.22, -0.16, 0.465],
    "table": [0.35, 0.0, 0.435],
    "drawer": [0.28, 0.0, 0.38],
}


def sim_command_to_plan_steps(
    command: dict,
    scene_coords: Optional[Dict[str, List[float]]] = None,
) -> List[PlanStep]:
    """Convert a prepared command dictionary into executable MuJoCo PlanStep primitives.

    Decomposes high-level macro actions ('pick', 'place') into physical 3D primitives
    (approach -> grasp -> lift, transport -> release), resolves 3D coordinates,
    and returns validated PlanStep instances.

    Args:
        command: Command dict containing 'sim_arm', 'action', 'obj', and optional 'target'.
                 If 'sim_arm' is not present, prepare_command() is run first.
        scene_coords: Optional dictionary overriding 3D object/target coordinates.

    Returns:
        List[PlanStep]: Sequence of executable simulation primitives.
    """
    cmd = command if "sim_arm" in command else prepare_command(command)
    arm = cmd["sim_arm"]
    action = cmd["action"]
    obj = cmd["obj"]
    target = cmd.get("target")

    coords_map = {**DEFAULT_OBJECT_COORDS, **(scene_coords or {})}

    steps: List[PlanStep] = []

    if action == "open":
        steps.append(PlanStep(action="open_drawer", arm=arm, object=obj, target=None))

    elif action == "pick":
        obj_target = coords_map.get(obj, [0.25, 0.0, 0.435])
        steps.append(PlanStep(action="approach", arm=arm, object=obj, target=obj_target))
        steps.append(PlanStep(action="grasp", arm=arm, object=obj, target=None))
        steps.append(PlanStep(action="lift", arm=arm, object=obj, target=None))

    elif action == "place":
        dest_target = coords_map.get(target or "table", [0.35, 0.0, 0.435])
        steps.append(PlanStep(action="transport", arm=arm, object=target or "table", target=dest_target))
        steps.append(PlanStep(action="release", arm=arm, object=obj, target=None))

    elif action == "pour":
        pour_target = coords_map.get(obj, [0.22, -0.16, 0.465])
        steps.append(PlanStep(action="pour", arm=arm, object=obj, target=pour_target))

    elif action == "handoff":
        other_arm = "right_arm" if arm == "left_arm" else "left_arm"
        handoff_coords = [0.25, 0.0, 0.48]
        steps.append(PlanStep(action="transport", arm=arm, object=obj, target=handoff_coords))
        steps.append(PlanStep(action="approach", arm=other_arm, object=obj, target=handoff_coords))
        steps.append(PlanStep(action="grasp", arm=other_arm, object=obj, target=None))
        steps.append(PlanStep(action="release", arm=arm, object=obj, target=None))

    return steps


def commands_to_plan_steps(
    commands: List[dict],
    scene_coords: Optional[Dict[str, List[float]]] = None,
) -> List[PlanStep]:
    """Convert a list of parser/prepared commands into a flat sequence of MuJoCo PlanSteps."""
    plan_steps: List[PlanStep] = []
    for cmd in commands:
        plan_steps.extend(sim_command_to_plan_steps(cmd, scene_coords=scene_coords))
    return plan_steps