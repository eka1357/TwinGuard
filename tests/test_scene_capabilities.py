import pytest

from planning.scene_capabilities import SCENE_OBJECTS, validate_scene_object


def test_plate_is_available():
    assert "plate" in SCENE_OBJECTS
    assert validate_scene_object("plate") is None


def test_table_is_available():
    assert "table" in SCENE_OBJECTS
    assert validate_scene_object("table") is None


@pytest.mark.parametrize("obj", ["mug", "drawer", "spoon"])
def test_unmodeled_object_is_rejected(obj):
    with pytest.raises(ValueError, match="not available"):
        validate_scene_object(obj)