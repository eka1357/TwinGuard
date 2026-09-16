"""Objects currently available in the MuJoCo scene."""

SCENE_OBJECTS = {"plate", "table"}


def validate_scene_object(obj: str) -> None:
    """Raise ValueError when an object is not modeled in the scene."""
    if obj not in SCENE_OBJECTS:
        raise ValueError(
            f"Object {obj!r} is not available in the current simulation scene. "
            f"Available objects: {sorted(SCENE_OBJECTS)}."
        )