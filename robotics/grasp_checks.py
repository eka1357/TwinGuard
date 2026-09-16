from robotics.reachability import gripper_to_plate_distance


def is_gripper_near_plate(
    sim,
    arm: str,
    *,
    max_distance: float = 0.08,
) -> bool:
    """Return True when the gripper is within max_distance of the plate.

    This is only a proximity check. It does not prove contact, grasp success,
    or object attachment.
    """
    if max_distance <= 0:
        raise ValueError("max_distance must be greater than zero")

    return gripper_to_plate_distance(sim, arm) <= max_distance


def require_gripper_near_plate(
    sim,
    arm: str,
    *,
    max_distance: float = 0.08,
) -> float:
    """Return distance if near enough, otherwise raise ValueError."""
    if max_distance <= 0:
        raise ValueError("max_distance must be greater than zero")

    distance = gripper_to_plate_distance(sim, arm)

    if distance > max_distance:
        raise ValueError(
            f"Gripper for {arm!r} is too far from the plate: "
            f"{distance:.4f} m > {max_distance:.4f} m."
        )

    return distance
