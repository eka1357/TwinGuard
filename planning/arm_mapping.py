ARM_MAP = {"A": "left_arm", "B": "right_arm"}


def parser_arm_to_sim_arm(arm: str) -> str:
    """Map a parser arm label to the corresponding simulator arm name."""
    if arm is None:
        raise ValueError("Invalid arm value: expected 'A' or 'B'.")

    normalized = str(arm).strip()
    if not normalized:
        raise ValueError("Invalid arm value: expected 'A' or 'B'.")

    key = normalized.upper()
    if key not in ARM_MAP:
        raise ValueError(f"Invalid arm value: {arm!r}. Expected 'A' or 'B'.")

    return ARM_MAP[key]