"""Executes validated planner commands against a TwinGuard simulator.

This module performs no inverse kinematics and invents no poses. It
only translates a small set of supported actions into direct gripper
calls on the simulator, for actions where that is sufficient (pick,
place, open). Unsupported actions (pour, handoff) are explicitly
rejected until a real motion/grasp strategy exists.
"""

from planning.scene_capabilities import validate_scene_object


def execute_command(sim, command: dict) -> None:
    """Execute one prepared command against the simulator."""
    if "sim_arm" not in command or not command["sim_arm"]:
        raise ValueError("Command is missing required 'sim_arm' field.")

    if "obj" not in command or not command["obj"]:
        raise ValueError("Command is missing required 'obj' field.")

    if "action" not in command or not command["action"]:
        raise ValueError("Command is missing required 'action' field.")

    validate_scene_object(command["obj"])

    action = command["action"]
    sim_arm = command["sim_arm"]

    if action == "open":
        sim.open_gripper(sim_arm)
    elif action == "pick":
        sim.close_gripper(sim_arm)
    elif action == "place":
        sim.open_gripper(sim_arm)
    elif action == "pour":
        raise NotImplementedError("pour is not yet implemented.")
    elif action == "handoff":
        raise NotImplementedError("handoff is not yet implemented.")
    else:
        raise ValueError(f"Unsupported action: {action!r}.")